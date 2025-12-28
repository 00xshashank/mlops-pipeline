from zenml import step, pipeline

import wandb
wandb.init()

@step(experiment_tracker="wandb_tracker")
def add(n1: int, n2: int) -> int:
    sum = n1 + n2
    wandb.log({"sum": sum})
    return sum

@step
def subtract(n1: int, n2: int) -> int:
    diff = n2 - n1
    wandb.log({"diff": diff})
    return diff

@pipeline
def pipelining(n1: int, n2: int) -> None:
    a = add(n1, n2)
    b = subtract(n1, n2)

if __name__ == "__main__":
    pipelining(1, 2)