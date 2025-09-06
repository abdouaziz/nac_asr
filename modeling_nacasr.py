from nacasr import NACASR
from nacasr import get_dataloader









if __name__ == "__main__":
    dataloader = get_dataloader("abdouaziiz/new_benchmark_wolof", batch_size=2, num_workers=4)
    for batch in dataloader:
        print(batch)
        break