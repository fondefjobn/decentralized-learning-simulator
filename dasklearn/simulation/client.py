from typing import Optional, Any, List

from dasklearn.model_trainer import AUGMENTATION_FACTOR_SIM
from dasklearn.simulation.bandwidth_scheduler import BWScheduler
from dasklearn.functions import *
from dasklearn.simulation.events import FINISH_TRAIN, Event, START_TRANSFER
from dasklearn.tasks.task import Task
from dasklearn.util.utils import get_random_hex_str


class BaseClient:

    def __init__(self, simulator, index: int):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.simulator = simulator
        self.index = index
        self.bw_scheduler = BWScheduler(self)
        self.other_nodes_bws: Dict[bytes, int] = {}

        self.simulated_speed: Optional[float] = None
        self.round: int = 0

        self.own_model: Optional[str] = None
        self.latest_task: Optional[str] = None  # Keep track of the latest task

    def client_log(self, msg: str):
        self.logger.info("[t=%.3f] %s", self.simulator.current_time, msg)

    def get_train_time(self) -> float:
        train_time: float = 0.0
        if self.simulated_speed:
            local_steps: int = self.simulator.settings.learning.local_steps
            batch_size: int = self.simulator.settings.learning.batch_size
            train_time = AUGMENTATION_FACTOR_SIM * local_steps * batch_size * (self.simulated_speed / 1000)
        return train_time

    def start_train(self, event: Event):
        """
        We started training. Schedule when the training has ended.
        """
        task_name = "train_%s" % get_random_hex_str(6)
        task = Task(task_name, "train", data={"model": self.own_model, "round": self.round, "peer": self.index})
        self.add_compute_task(task)

        finish_train_event = Event(event.time + self.get_train_time(), self.index, FINISH_TRAIN,
                                   data={"model": task_name})
        self.simulator.schedule(finish_train_event)

    def send_model(self, to: int, model: str, metadata: Optional[Dict[Any, Any]] = None) -> None:
        metadata = metadata or {}
        event_data = {"from": self.index, "to": to, "model": model, "metadata": metadata}
        start_transfer_event = Event(self.simulator.current_time, self.index, START_TRANSFER, data=event_data)
        self.simulator.schedule(start_transfer_event)

    def start_transfer(self, event: Event):
        """
        We started a transfer operation. Compute how much time it takes to complete the transfer and schedule the
        completion of the transfer.
        """
        receiver_scheduler: BWScheduler = self.simulator.clients[event.data["to"]].bw_scheduler
        self.bw_scheduler.add_transfer(receiver_scheduler, self.simulator.model_size, event.data["model"],
                                       event.data["metadata"])

    def finish_outgoing_transfer(self, event: Event):
        """
        An outgoing transfer has finished.
        """
        self.bw_scheduler.on_outgoing_transfer_complete(event.data["transfer"])

    def aggregate_models(self, models: List[str], weights: List[float] = None) -> str:
        task_name = "agg_%s" % get_random_hex_str(6)
        data = {"models": models, "round": self.round, "peer": self.index}
        if weights:
            data["weights"] = weights
        task = Task(task_name, "aggregate", data=data)
        self.add_compute_task(task)
        return task_name

    def add_compute_task(self, task: Task):
        self.simulator.workflow_dag.tasks[task.name] = task
        self.latest_task = task

        # Link inputs/outputs of the task
        if (task.func == "train" and task.data["model"] is not None) or task.func == "test":
            preceding_task: Task = self.simulator.workflow_dag.tasks[task.data["model"]]
            preceding_task.outputs.append(task)
            task.inputs.append(preceding_task)
        elif task.func == "aggregate":
            for model_name in task.data["models"]:
                preceding_task: Task = self.simulator.workflow_dag.tasks[model_name]
                preceding_task.outputs.append(task)
                task.inputs.append(preceding_task)
