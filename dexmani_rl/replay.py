import torch


class Replay:
    def __init__(self, capacity):
        self.capacity, self.position, self.size = capacity, 0, 0
        self.data = {}

    def add(self, **transition):
        for key, value in transition.items():
            value = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
            if key not in self.data:
                self.data[key] = torch.empty((self.capacity, *value.shape), dtype=torch.float32)
            self.data[key][self.position].copy_(value)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device, *, fields):
        indices = torch.randint(self.size, (batch_size,))
        return {key: self.data[key][indices].to(device) for key in fields}

    def state_dict(self):
        data = self.data
        if self.size < self.capacity:
            # torch.save writes the backing storage of a view, including unused rows.
            data = {key: value[:self.size].clone() for key, value in data.items()}
        return dict(capacity=self.capacity, position=self.position, size=self.size, data=data)

    def load_state_dict(self, state):
        self.capacity, self.position, self.size = (state[k] for k in ('capacity', 'position', 'size'))
        self.data = {k: torch.empty((self.capacity, *v.shape[1:])) for k, v in state['data'].items()}
        for k, v in state['data'].items():
            self.data[k][:self.size].copy_(v)
