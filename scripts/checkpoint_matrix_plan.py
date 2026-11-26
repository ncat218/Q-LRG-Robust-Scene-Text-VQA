"""CPU-only provenance and exact experiment plan for the final checkpoint matrix."""

import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def resolve_training(root, view, method=None):
    """Use completed all-train refit, or explicitly retain the zero baseline.

    Never silently substitute selection/latest, a partly trained refit or the
    highest official-validation score for the checkpoint selected on val_inner.
    """
    root = Path(root)
    stage = 's1_a2_convergence' if method else 's1_convergence'
    complete = read(root / 'COMPLETE.json')
    if complete['stage'] != stage or set(complete['views']) != {'full', 'crop'}:
        raise ValueError('Training group must have completed both views')
    directory = root / view
    if digest(directory / 'summary.json') != complete['summaries'][view]:
        raise ValueError('Training group summary checksum changed')
    if method:
        directory = directory / method
    summary = read(directory / 'summary.json')
    marker = read(directory / 'COMPLETE.json')
    if digest(directory / 'summary.json') != marker['summary_sha256']:
        raise ValueError('Training method/view summary changed')
    binding = read(directory / 'run.json')
    expected = hashlib.sha256(json.dumps(binding, ensure_ascii=False, sort_keys=True,
                                        allow_nan=False).encode()).hexdigest()
    if expected != marker['binding']:
        raise ValueError('Training source binding changed')
    if method and summary.get('a2_unchanged') is not True:
        raise ValueError('The supposedly fixed A2 reader changed during training')
    selected = summary['selection']
    selected_path = directory / 'selection' / selected['checkpoint']
    if digest(selected_path) != selected['checkpoint_sha256']:
        raise ValueError('Selected checkpoint checksum changed')
    if read(directory / 'selection' / 'COMPLETE.json') != selected:
        raise ValueError('Selection completion marker differs')
    if selected['binding'] != expected:
        raise ValueError('Selection belongs to another training source')
    visits = selected['selection']['best_examples']
    refit = summary['refit']
    record = {'training_run': str(root), 'view': view, 'method': method or 'restoration',
              'training_binding': binding, 'selected_examples': visits,
              'selection_checkpoint_sha256': selected['checkpoint_sha256'],
              'selection_source': str(selected_path), 'selection_binding': selected['binding'],
              'summary_sha256': digest(directory / 'summary.json')}
    if visits == 0:
        if not isinstance(refit, dict) or 'skipped' not in refit:
            raise ValueError('Zero-baseline selection must explicitly skip refit')
        return {**record, 'source': None, 'sha256': None, 'baseline_kept': True,
                'reason': refit['skipped']}
    if not refit or 'skipped' in refit:
        raise ValueError('Winning trained checkpoint requires completed all-train refit')
    path = directory / 'refit' / refit['checkpoint']
    if digest(path) != refit['checkpoint_sha256']:
        raise ValueError('Refit checkpoint checksum changed')
    if read(directory / 'refit' / 'COMPLETE.json') != refit:
        raise ValueError('Refit completion marker differs')
    refit_binding = {**binding, 'selected_examples': visits,
                     'selected_checkpoint_sha256': selected['checkpoint_sha256'],
                     'refit_visits': refit['examples']}
    if not method:
        refit_binding['refit_rule'] = 'same selected fractional epochs over all official_train'
    expected_refit = hashlib.sha256(json.dumps(refit_binding, ensure_ascii=False,
                                             sort_keys=True, allow_nan=False).encode()).hexdigest()
    if refit['binding'] != expected_refit:
        raise ValueError('Refit binding does not match the selected checkpoint/duration')
    return {**record, 'source': str(path), 'sha256': refit['checkpoint_sha256'],
            'checkpoint_binding': refit['binding'], 'examples': refit['examples'],
            'refit_epochs': refit['epochs_completed'], 'baseline_kept': False}


def module_plan(layout, modules, has_box):
    """Preserve which module was requested even when crop falls back to full."""
    if layout == 'full':
        return ('full',), modules
    if layout == 'crop':
        return (('crop',) if has_box else ('full',)), modules
    if layout != 'fusion':
        raise ValueError('Unknown input layout')
    return (('full', 'crop'), modules) if has_box else (('full',), (modules[0],))


def arm_name(reader, condition, budget, layout, modules):
    return '__'.join((reader, condition, budget, layout, '+'.join(m or 'none' for m in modules)))


def specifications(profiles):
    """Alias identical baselines/zero modules; do not run redundant generations."""
    canonical, table = {}, []
    for profile, settings in profiles.items():
        for reader in settings['readers']:
            for condition in ('CLEAN', 'LR4'):
                for budget in ('fair576', 'native'):
                    for layout in settings.get('layouts', ('full', 'crop', 'fusion')):
                        views = ('full', 'crop') if layout == 'fusion' else (layout,)
                        toggles = ((0, 0), (1, 0), (0, 1), (1, 1)) if layout == 'fusion' else ((0,), (1,))
                        for flags in toggles:
                            modules = tuple(settings[v] if flag else None for v, flag in zip(views, flags))
                            name = arm_name(reader, condition, budget, layout, modules)
                            canonical[name] = {'arm': name, 'reader': reader, 'condition': condition,
                                               'budget': budget, 'layout': layout, 'modules': list(modules)}
                            table.append({'profile': profile, 'reader': reader, 'condition': condition,
                                          'budget': budget, 'layout': layout,
                                          'flags': ''.join(map(str, flags)), 'arm': name})
    return list(canonical.values()), table


def comparison_pairs(table):
    index = {(r['profile'], r['reader'], r['condition'], r['budget'], r['layout'], r['flags']): r['arm']
             for r in table}
    pairs = set()
    for row in table:
        p, r, c, b, layout, flags = (row[k] for k in ('profile', 'reader', 'condition', 'budget', 'layout', 'flags'))
        target = row['arm']
        keys = [(p, r, c, b, layout, '0' * len(flags)), (p, r, c, b, 'full', '0'),
                ('old11305', r, c, b, layout, flags)]
        if p == 'new_A2_restoration':
            keys += [('new_A0', r, c, b, layout, flags), ('new_A2_ce_only', r, c, b, layout, flags)]
        if layout == 'fusion' and flags == '11':
            # Isolate adding crop restoration when full restoration is already on,
            # and vice versa; both-on versus both-off is not sufficient.
            keys += [(p, r, c, b, layout, '10'), (p, r, c, b, layout, '01')]
        if p == 'mixed_old_full_A2_crop':
            keys += [('mixed_old_full_A0_crop', r, c, b, layout, flags)]
        if r == 'A2':
            keys += [(p, 'A0', c, b, layout, flags)]
        for key in keys:
            if key in index and index[key] != target:
                pairs.add((index[key], target))
    return sorted(pairs)
