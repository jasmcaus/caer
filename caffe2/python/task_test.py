import unittest
from caffe2.python import task


class TestTask(unittest.TestCase):
    def testRepr(self):
        cases = [
            (task.Cluster(), "Cluster(nodes=[], node_kwargs={})"),
            (task.Node(), "Node(name=local, kwargs={})"),
            (
                task.TaskGroup(),
                "TaskGroup(tasks=[], workspace_type=None, remote_nets=[])",
            ),
            (task.TaskOutput([]), "TaskOutput(names=[], values=None)"),
            (task.Task(), "Task(name=local/task, node=local, outputs=[])"),
            (task.SetupNets(), "SetupNets(init_nets=None, exit_nets=None)"),
        ]
        for obj, want in cases:
            self.assertEqual(obj.__repr__(), want)

    def testEffectlessRepr(self):
        task_group = task.TaskGroup()
        _repr = task_group.__repr__()
        self.assertFalse(task_group._already_used)
