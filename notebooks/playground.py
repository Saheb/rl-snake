import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch

    return (torch,)


@app.cell
def _(torch):
    x = torch.tensor([1,2,3])
    y = torch.tensor([4,5,6])
    torch.add(x, y)
    return


@app.cell
def _():
    import random

    return (random,)


@app.cell
def _(random):
    t = random.sample([(10,20), (3,4)], k=2)
    return


@app.cell
def _(unzip):
    unzip
    return


if __name__ == "__main__":
    app.run()
