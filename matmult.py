import tensorflow as tf
import numpy as np

# Initialize MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Create MirroredStrategy
strategy = tf.distribute.MirroredStrategy()
print("Num replicas: ", strategy.num_replicas_in_sync)
print("Replica ID: ", strategy.cluster_resolver.task_id)

def dist_matmul(A, B):
    m, n = A.shape
    _, p = B.shape

    with strategy.scope():
        # Create variables to store results of each process
        C_values = tf.Variable(tf.zeros([m//size, p//size]), dtype=tf.float32)
        C = tf.concat([C_values for _ in range(size)], axis=0)

        # Split matrices A and B and distribute them to different processes
        A_split = tf.split(A, size, axis=0)
        B_split = tf.split(B, size, axis=1)

        # Multiply the split matrices on each process
        C_local = tf.matmul(A_split[rank], B_split[rank])

        # All-reduce to sum the results across all processes
        C = tf.distribute.get_replica_context().all_reduce(C_local, tf.distribute.ReduceOp.SUM)

        # Return the result
        return C

# Set the dimensions of the matrices
m, n, p = 2000, 2000, 2000

# Generate random matrices
A = np.random.rand(m, n).astype(np.float32)
B = np.random.rand(n, p).astype(np.float32)

# Multiply matrices
C = dist_matmul(A, B)

# Print results from rank 0
if rank == 0:
    print(C)
