import torch

# Class that applies a normal forward pass. 
# The resulting gradients will be scaled by a factor of 0.5.

class BackPropagationGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Normal forward pass
        ctx.save_for_backward(input)
        output = input.clone()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Scaled backpropagation
        input, = ctx.saved_tensors
        grad_input = grad_output.clone() * 0.5
        return grad_input


# Usage example
# x = torch.randn(3, 5, requires_grad=True)
# y = BackPropagationGate.apply(x)
# z = y.mean()
# z.backward()
# print(x.grad)