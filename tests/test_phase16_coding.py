#!/usr/bin/env python3
"""Check the frozen coding tasks against their registered fixtures."""
import hashlib
import json
from pathlib import Path
import subprocess
import unittest


TASKS_PATH = Path(__file__).resolve().parents[1] / 'bench/eval/phase16-coding.json'
IDS = [
    'py_eval_sum_range', 'py_write_is_even', 'c_fix_loop_bound', 'c_sizeof_string',
    'py_write_squares', 'py_eval_fib', 'py_fix_default_arg', 'py_eval_unique_keep_order',
]


def score(task, content):
    content += '\n'

    def matches(pattern):
        result = subprocess.run(['grep', '-Ei', '-e', pattern], input=content,
                                text=True, capture_output=True)
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr)
        return result.returncode == 0

    required = [matches(pattern) for pattern in task['required_patterns']]
    forbidden = [matches(pattern) for pattern in task['forbidden_patterns']]
    words = subprocess.run(['wc', '-w'], input=content, text=True,
                           capture_output=True, check=True)
    return (all(required) and not any(forbidden) and
            (task['max_words'] <= 0 or int(words.stdout) <= task['max_words']))


FIXTURES = {
    'py_eval_sum_range': (
        ['10', '10.', '10!'],
        ['15', 'The answer is 10', '0+1+2+3+4=10'],
    ),
    'py_write_is_even': (
        ['def is_even(n):\n    return n % 2 == 0',
         'def is_even(n):\n    return n%2==0',
         'def is_even(n):\n    return (n % 2) == 0',
         'def is_even(n):\n    return not n % 2',
         'def is_even(n):\n    return (n & 1) == 0',
         '```python\ndef is_even(n):\n    return n%2==0\n```'],
        ['def is_even(n):\n    return n & 1 == 0',
         'def is_even(n):\n    return n % 2 != 1',
         'def is_odd(n):\n    return n % 2 == 0',
         'def is_even(n):\n    print(n)\n    return n % 2 == 0',
         'def is_even(n):\n    if n % 2 == 0: return True'],
    ),
    'c_fix_loop_bound': (
        ['for (int i = 1; i <= n; i++)', 'for(int i=1;i<=n;i++)',
         'for (int i = 1; i < n + 1; i++)', 'for (int i = 1; i < (n + 1); ++i)'],
        ['for (int i = 1; i < n; i++)', 'for (int i = 0; i <= n; i++)', 'i <= n'],
    ),
    'c_sizeof_string': (['3', '3.'], ['2', 'sizeof is 3']),
    'py_write_squares': (
        ['def squares(n):\n    return [i**2 for i in range(n)]',
         'def squares(n):\n    return [i ** 2 for i in range(n)]',
         'def squares(n):\n    return [x * x for x in range(n)]',
         'def squares(n):\n    return [i*i for i in range(n)]',
         '```python\ndef squares(n):\n    return [i**2 for i in range(n)]\n```'],
        ['def squares(n):\n    out = []\n    for i in range(n):\n        out.append(i*i)\n    return out',
         'def squares(n):\n    return [i**2 for i in range(n+1)]',
         'def square(n):\n    return [i**2 for i in range(n)]',
         'def squares(n):\n    return [i**2 for x in range(n)]',
         'def squares(n):\n    return [i * x for i in range(n)]'],
    ),
    'py_eval_fib': (['8', '8.'], ['13', 'fib(6)=8']),
    'py_fix_default_arg': (
        ['def add_item(item, acc=None):', 'def add_item(item,acc=None):'],
        ['def add_item(item, acc=[]):', 'def add_item(item, acc=()):',
         'def add_item(item, acc=None):\n    if acc is None:\n        acc = []\n    return acc'],
    ),
    'py_eval_unique_keep_order': (
        ['[1, 2, 3]', '[1,2, 3]', '[1,2,3]'],
        ['[1, 2, 1, 3, 2]', '[3, 2, 1]', '1, 2, 3'],
    ),
}


class Phase16CodingTests(unittest.TestCase):
    def test_frozen_file(self):
        data = TASKS_PATH.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(),
                         '5f0bd6e551223507561909d911a485c65af3a8358c60d07e098b11e84b8e89f1')
        self.assertTrue(data.endswith(b'\n'))
        self.assertFalse(data.endswith(b'\n\n'))
        self.assertNotIn(b'\r', data)

    def test_schema(self):
        tasks = json.loads(TASKS_PATH.read_bytes())
        self.assertIsInstance(tasks, list)
        self.assertEqual(len(tasks), 8)
        for task in tasks:
            self.assertIsInstance(task, dict)
            self.assertTrue({'id', 'messages', 'max_tokens', 'required_patterns',
                             'forbidden_patterns', 'max_words'} <= task.keys())
            self.assertLessEqual(task['max_tokens'], 160)
        self.assertEqual([task['id'] for task in tasks], IDS)

    def test_fixtures(self):
        tasks = {task['id']: task for task in json.loads(TASKS_PATH.read_bytes())}
        for task_id, (passes, fails) in FIXTURES.items():
            for expected, replies in ((True, passes), (False, fails)):
                for reply in replies:
                    with self.subTest(task_id=task_id, fixture=reply):
                        self.assertEqual(score(tasks[task_id], reply), expected)


if __name__ == '__main__':
    unittest.main()
