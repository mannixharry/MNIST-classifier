from pathlib import Path

from data import load_mnist
from model import init_params
from train import train, evaluate

DATA_DIR = Path(__file__).parent / "data"

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist(DATA_DIR)
    print(f"train {X_train.shape}  val {X_val.shape}  test {X_test.shape}")
    
    params = init_params(n_in=784, n_hidden=128, n_out=10, seed=0)
    
    train(params, X_train, y_train, X_val, y_val,
          epochs=100, batch_size=64, lr=0.1, seed=0)
    
    test_loss, test_acc = evaluate(params, X_test, y_test)
    print(f"\ntest  loss {test_loss:.4f}  acc {test_acc:.4f}")
    
if __name__ == "__main__":
    main()