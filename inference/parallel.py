import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def compute_many(pipeline, hashes, n_cpus=1, *args, **kwargs):
    """Almost copied from https://github.com/JelleAalbers/blueice/blob/master/blueice/parallel.py#L47"""
    if n_cpus != 1:
        pool = ProcessPoolExecutor(max_workers=n_cpus)

        futures = []
        for h in hashes:
            futures.append(pool.submit(pipeline, *args, hash=h, **kwargs))

        # Wait for the futures to complete; give a progress bar
        with tqdm(total=len(futures), desc='Computing on %d cores' % n_cpus) as pbar:
            while len(futures):
                _done_is = []
                for f_i, f in enumerate(futures):
                    if f.done():
                        f.result()     # Raises exception on error
                        _done_is.append(f_i)
                        pbar.update(1)
                futures = [f for f_i, f in enumerate(futures) if not f_i in _done_is]
                time.sleep(0.1)
    else:
        for h in tqdm(hashes, desc='Computing on one core'):
            pipeline(*args, hash=h, **kwargs)
