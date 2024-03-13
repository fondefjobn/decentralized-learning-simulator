from typing import Dict, List, Set, Tuple, Any
from collections import Counter

from networkx import DiGraph
from torch import nn
import networkx as nx

from dasklearn.tasks.task import Task


class WorkflowDAG:

    def __init__(self):
        self.tasks: Dict[str, Task] = {}  # Workflow DAG

    def get_source_tasks(self) -> List[Task]:
        """
        Get the tasks without any input.
        """
        return [task for task in self.tasks.values() if not task.inputs]

    def get_sink_tasks(self) -> List[Task]:
        """
        Get the tasks without any output.
        """
        return [task for task in self.tasks.values() if not task.outputs]

    def serialize(self) -> List[Dict]:
        """
        Serialize the DAG.
        """
        return [task.to_json_dict() for task in self.tasks.values()]

    @classmethod
    def unserialize(cls, serialized_tasks):
        dag = WorkflowDAG()
        for serialized_task in serialized_tasks:
            task: Task = Task.from_json_dict(serialized_task)
            dag.tasks[task.name] = task

        # Now that we created all Task objects, fix the inputs and outputs
        for serialized_task in serialized_tasks:
            task = dag.tasks[serialized_task["name"]]
            for input_task_name in serialized_task["inputs"]:
                task.inputs.append(dag.tasks[input_task_name])
            for output_task_name in serialized_task["outputs"]:
                task.outputs.append(dag.tasks[output_task_name])

        return dag

    def print_tasks(self):
        for task in self.tasks.values():
            print("Task %s, in: %s, out: %s" % (task, task.inputs, task.outputs))

    @staticmethod
    def count_models(d, model_hashes: Set[int]):
        if isinstance(d, nn.Module):
            model_hashes.add(id(d))
        elif isinstance(d, dict):
            for item in d.values():
                WorkflowDAG.count_models(item, model_hashes)
        elif isinstance(d, list):
            for item in d:
                WorkflowDAG.count_models(item, model_hashes)

        return model_hashes

    def get_num_models(self) -> int:
        """
        Compute recursively the number of models in all tasks.
        """
        model_hashes = set()
        WorkflowDAG.count_models([d.data for d in self.tasks.values() if d.data is not None], model_hashes)
        return len(model_hashes)

    def to_nx(self, max_size: int) -> tuple[DiGraph, dict[str, tuple[int | int]]]:
        graph = nx.DiGraph()
        layer = {}
        at_layer = Counter()
        position = {}
        for task in self.tasks.values():
            if max_size <= 0:
                break
            if len(task.inputs) == 0:
                layer[task.name] = 0
                graph.add_node(task.name)
            else:
                layer[task.name] = min(map(lambda x: layer[x.name], task.inputs)) - 1
                for inp in task.inputs:
                    graph.add_edge(task.name, inp.name)
            max_size -= 1
            position[task.name] = at_layer[layer[task.name]], layer[task.name]
            if at_layer[layer[task.name]] == 0:
                at_layer[layer[task.name]] = 1
            elif at_layer[layer[task.name]] > 0:
                at_layer[layer[task.name]] *= -1
            else:
                at_layer[layer[task.name]] *= -1
                at_layer[layer[task.name]] += 1
        return graph, position
