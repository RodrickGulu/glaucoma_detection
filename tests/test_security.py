import os
import tempfile
import unittest

from flask import Flask

import db


class SecurityTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, 'users.db')
        db.DATABASE = self.db_path

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.app = Flask(__name__, root_path=project_root)
        self.app.config['TESTING'] = True
        self.app.app_context().push()
        db.init_db()

    def tearDown(self):
        try:
            db.close_db()
        finally:
            while self.app.app_context()._cv_tokens:
                self.app.app_context().pop()
            self.temp_dir.cleanup()

    def test_passwords_are_hashed_before_storage(self):
        db.add_user('Alice Example', 'alice', 'S3cret!')
        saved_hash = db.get_db().execute(
            'SELECT password FROM users WHERE username = ?', ('alice',)
        ).fetchone()[0]

        self.assertNotEqual(saved_hash, 'S3cret!')
        self.assertTrue(db.authenticate('alice', 'S3cret!'))
        self.assertFalse(db.authenticate('alice', 'wrong-password'))


if __name__ == '__main__':
    unittest.main()
