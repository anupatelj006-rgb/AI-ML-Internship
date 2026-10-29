import torch


def tensor_demo():
    print("=== CREATION ===")
    a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    print("a =\n", a, "\nshape:", tuple(a.shape))
    print("zeros:\n", torch.zeros(2, 3))
    print("ones:\n", torch.ones(2, 3))
    print("random:\n", torch.rand(2, 3))
    print("range:", torch.arange(6))

    print("\n=== RESHAPING ===")
    print("reshape to 3x2:\n", a.reshape(3, 2))
    print("flatten to one row:", a.reshape(-1))
    print("add a batch dimension:", tuple(a.unsqueeze(0).shape))
    print("transpose:\n", a.T)

    print("\n=== MULTIPLICATION ===")
    b = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
    print("element-wise a * b:\n", a * b)
    print("matrix multiply a @ b.T (2x3 times 3x2 = 2x2):\n", a @ b.T)
    print("multiply by a number a * 10:\n", a * 10)

    print("\n=== INDEXING ===")
    print("first row a[0]:", a[0])
    print("second column a[:, 1]:", a[:, 1])
    print("one item a[1, 2]:", a[1, 2])
    print("slice a[:, :2]:\n", a[:, :2])
    print("values bigger than 3, a[a > 3]:", a[a > 3])