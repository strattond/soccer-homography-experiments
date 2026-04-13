import queue

print_queue     = queue.Queue(maxsize=64)
annotate_queue  = queue.Queue(maxsize=128)
decode_queue    = queue.Queue(maxsize=256)
timing_queue    = queue.Queue(maxsize=16)