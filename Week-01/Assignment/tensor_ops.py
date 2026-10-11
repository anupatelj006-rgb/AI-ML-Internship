"""Part 2: tensor creation, reshaping, multiplication and indexing."""
import torch


def _section(title):
    print(f"\n--- {title} ---")


def demo_creation():
    _section("Creation")
    a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    b = torch.rand(3, 2)
    zeros = torch.zeros(2, 3)
    steps = torch.arange(0, 10, 2)
    print("a =", a.tolist(), "-> shape", tuple(a.shape))
    print("random b -> shape", tuple(b.shape))
    print("zeros -> shape", tuple(zeros.shape))
    print("arange(0,10,2) =", steps.tolist())
    return a, b


def demo_reshaping(a):
    _section("Reshaping")
    print("reshape(3, 2):", tuple(a.reshape(3, 2).shape))
    print("flatten with view(-1):", tuple(a.view(-1).shape))
    print("unsqueeze(0):", tuple(a.unsqueeze(0).shape))
    print("transpose .T:", tuple(a.T.shape))


def demo_multiplication(a, b):
    _section("Multiplication")
    ones = torch.ones_like(a)
    print("element-wise a * ones:", tuple((a * ones).shape))
    print("scalar a * 10:", (a * 10).tolist())
    product = a @ b
    print("matrix product a @ b: (2x3) @ (3x2) ->", tuple(product.shape))


def demo_indexing(a):
    _section("Indexing")
    print("a[0, 1]   :", a[0, 1].item())
    print("a[1]      :", a[1].tolist())
    print("a[:, 0]   :", a[:, 0].tolist())
    print("a[:, 1:]  :", a[:, 1:].tolist())
    print("a[a > 2]  :", a[a > 2].tolist())


def run_all():
    print("=" * 50)
    print("PART 2: TENSOR OPERATIONS")
    print("=" * 50)
    a, b = demo_creation()
    demo_reshaping(a)
    demo_multiplication(a, b)
    demo_indexing(a)