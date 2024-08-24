from multiprocessing import Process

from petsc4py import PETSc


def solve_equation(rank):
    # PETScのMPI環境を初期化
    comm = PETSc.COMM_WORLD
    rank = comm.getRank()

    # 各プロセスで異なる行列Aとベクトルbを定義
    n = rank + 2
    A = PETSc.Mat().create(comm=comm)
    A.setSizes([n, n])
    A.setFromOptions()
    A.setUp()

    for i in range(n):
        A[i, i] = 2.0 + rank
        if i > 0:
            A[i, i - 1] = -1.0
        if i < n - 1:
            A[i, i + 1] = -1.0

    A.assemble()

    b = PETSc.Vec().create(comm=comm)
    b.setSizes(n)
    b.setFromOptions()

    for i in range(n):
        b[i] = 1.0 + rank

    b.assemble()

    x = b.duplicate()
    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOperators(A)
    ksp.setFromOptions()

    ksp.solve(b, x)

    x_values = x.getArray()
    print(f"Process {rank}: Solution x = {x_values}")


if __name__ == "__main__":
    # PETScとMPIの初期化
    PETSc._initialize()

    # 複数のプロセスを作成
    num_processes = 4
    processes = []

    for rank in range(num_processes):
        p = Process(target=solve_equation, args=(rank,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # PETScの終了
    PETSc.Finalize()
