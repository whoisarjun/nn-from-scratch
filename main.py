import pseudomath as math
import pseudotorch as torch

def print_progress(j, total, bar_length=30):
    fraction = j / total
    filled_len = int(bar_length * fraction)
    bar = '█' * filled_len + '-' * (bar_length - filled_len)
    print(f'\r|{bar}| {i}/{total} ({fraction*100:.1f}%)', end='', flush=True)

model = torch.Network([
    torch.Layer(28*28, 16, activation='relu'),
    torch.Layer(16, 16, activation='relu'),
    torch.Layer(16, 10, activation='softmax')
])

total_items = 10000
epochs = 30

for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    loader = torch.DataLoader('data/train')  # reset iterable data loader
    for i, (target, matrix) in enumerate(loader):
        print_progress(i+1, total_items)
        result = model.forward(matrix)
        target_vector = math.Matrix.const(0., (10, 1))
        target_vector[target, 0] = 1.
        model.backpropagate(result, target_vector)
    print(f'\nCost: {model.cost:.4f}\n')

print('\n\n------------------ FINAL WEIGHTS ------------------\n')
for layer in model.layers:
    print(layer)
    print(f'Weights: {layer.weights}')
    print(f'Bias: {layer.bias}\n\n')
