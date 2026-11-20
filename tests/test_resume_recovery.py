import errno
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.resume_recovery import failure_record,io_wrapper,safe_write


class RecoveryTests(unittest.TestCase):
    def test_transient_io_failure_retries_without_losing_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);target=root/'latest.pt';target.write_bytes(b'old');calls=[]
            def writer(path,data):
                calls.append(1)
                if len(calls)==1:raise OSError(errno.ESTALE,'stale handle')
                return safe_write(path,data)
            with patch('scripts.resume_recovery.time.sleep'):
                io_wrapper(writer,root/'journal')(target,b'new')
            self.assertEqual(target.read_bytes(),b'new');self.assertEqual(len(calls),2)

    def test_quota_failure_keeps_old_and_journals_emergency_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);target=root/'latest.pt';target.write_bytes(b'old');calls=[]
            def fail(path,data):
                calls.append(1);raise OSError(errno.EDQUOT,'quota exceeded')
            with patch('scripts.resume_recovery.time.sleep'):
                with self.assertRaises(OSError):io_wrapper(fail,root/'journal')(target,b'emergency')
            self.assertEqual(target.read_bytes(),b'old');self.assertEqual(len(calls),3)
            j=json.loads((root/'journal/emergency-checkpoint.json').read_text())
            self.assertEqual(Path(j['backup']).read_bytes(),b'emergency')

    def test_failure_journal_records_type_message_and_frame_without_source_lookup(self):
        try:raise ValueError('diagnostic')
        except ValueError as error:j=failure_record(error)
        self.assertEqual(j['error_type'],'ValueError');self.assertEqual(j['message'],'diagnostic')
        self.assertTrue(j['frames']);self.assertEqual(j['frames'][-1]['function'],
                    'test_failure_journal_records_type_message_and_frame_without_source_lookup')


if __name__=='__main__':unittest.main()
