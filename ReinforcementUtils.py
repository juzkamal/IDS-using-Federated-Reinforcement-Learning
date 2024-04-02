import torch
import torch.nn as nn
import numpy as np
import random
from operator import itemgetter

from MyUtils import get_tensor_loader
from Args import args






class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x.float()))  # Convert input to float
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.feature_layer = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())

        return qvals
    
class DarkexperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity

        self.buffer = []

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer.pop(0)  # Remove the oldest experience
            self.buffer.append(transition)  # Add the new experience


    def sample(self, batch_size, w=0.2):
#             losses = np.array([experience[5] for experience in self.buffer])
            losses = np.array([experience[5].detach().numpy() for experience in self.buffer])
            loss_powers = losses ** w
            probs = loss_powers / np.sum(loss_powers)
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
            samples = [self.buffer[idx] for idx in indices]
            return samples
#


class DoubleDQNAgent:
    def __init__(self, state_size, action_size, memory_capacity=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = QNetwork(state_size, action_size)
        self.target_model = QNetwork(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = DarkexperienceReplayMemory(memory_capacity)

#     def act(self, state):
        
#         if np.random.rand() <= self.epsilon:
            
#             return random.randrange(self.action_size)
#         with torch.no_grad():
#             state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
#             q_values = self.model(state)
#             return torch.argmax(q_values).item()
    def act(self, state):#bahvior policy bri to take action.

            if np.random.rand() <= self.epsilon:
                return np.random.randint(self.action_size, size=len(state))  # Generate random actions for each state
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float).view(-1, 1)  # Adjust the dimensions
                q_values = self.model(state_tensor)
                actions = torch.argmax(q_values, dim=1).numpy()
                return actions




    def remember(self, state, action, reward, next_state, done, loss, logits):
        transition = (state, action, reward, next_state, done, loss ,logits)
#         priority = (abs(loss) + 1e-6) ** 0.6  # prioritization based on the loss
        self.memory.push(transition)





    def replay(self, batch_size, beta):
        samples = self.memory.sample(batch_size, beta)
        for sample in samples:
            state, action, reward, next_state, done, loss , logits = sample
            # Update DQN weights using Bellman's equation
#             if next_state is not None:
#                 target = reward + gamma * torch.max(agent.target_model(torch.tensor(next_state)))
#             else:
#                 target = torch.tensor(reward)
                
                
                
            if next_state is not None:
                target = reward + gamma * torch.max(agent.target_model(torch.tensor(next_state)) , dim= 1)[0]
            else:
                target = torch.tensor(reward)    
                
                
#             qqvalues= agent.model(torch.tensor(state))
#             q_value, mm = torch.max(qqvalues, dim = 1) 
#             loss = ((torch.abs(target - q_value))**0.5).sum()   
                
#             q_values = self.model(torch.tensor(state))

#             selected_rows = q_values[torch.arange(q_values.size(0)), action]
    
#             error = torch.abs(selected_rows - target)
#             darkexperienceloss= ((torch.abs(logits-qqvalues))**2)
#             darkexperienceloss= torch.sum(darkexperienceloss, dim= 1)
#             dd= darkexperienceloss.sum() 
            
#             loss = loss+ 0.2 * darkexperienceloss
            qqvalues = agent.model(torch.tensor(state))
            q_value, _ = torch.max(qqvalues, dim=1)
            loss3 = ((torch.abs(target - q_value)) ** 2).sum()
            
            darkexperienceloss = ((torch.abs(logits - qqvalues)) ** 2).sum()
            loss2 = 0.2 * darkexperienceloss 
            loss = loss3 + loss2  # Add the two losses
            loss = loss3  # Compute the sum of the combined losses

           
            #print(loss)# Adding to the loss
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
# Backward pass
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)  # Set retain_graph=True to retain the computational graph
            self.optimizer.step()

#         priorities = [abs(sample[5]) + 1e-6 for sample in samples]
#         self.memory.update_priorities(range(len(samples)), priorities)

#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay

    def CopyCurrent_and_targetDQNS(self):
        self.target_model.load_state_dict(self.model.state_dict())

class DuelingDDQNAgent:
    def __init__(self, state_size, action_size, memory_capacity=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DuelingDQN(state_size, action_size)
        self.target_model = DuelingDQN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = DarkexperienceReplayMemory(memory_capacity)

#     def act(self, state):
        
#         if np.random.rand() <= self.epsilon:
            
#             return random.randrange(self.action_size)
#         with torch.no_grad():
#             state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
#             q_values = self.model(state)
#             return torch.argmax(q_values).item()
    def act(self, state):#bahvior policy bri to take action.

            if np.random.rand() <= self.epsilon:
                return np.random.randint(self.action_size, size=len(state))  # Generate random actions for each state
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float).view(-1, 1)  # Adjust the dimensions
                q_values = self.model(state_tensor)
                actions = torch.argmax(q_values, dim=1).numpy()
                return actions




    def remember(self, state, action, reward, next_state, done, loss, logits):
        transition = (state, action, reward, next_state, done, loss ,logits)
#         priority = (abs(loss) + 1e-6) ** 0.6  # prioritization based on the loss
        self.memory.push(transition)





    def replay(self, batch_size, beta):
        samples = self.memory.sample(batch_size, beta)
        for sample in samples:
            state, action, reward, next_state, done, loss , logits = sample
            # Update DQN weights using Bellman's equation
#             if next_state is not None:
#                 target = reward + gamma * torch.max(agent.target_model(torch.tensor(next_state)))
#             else:
#                 target = torch.tensor(reward)
                
                
                
            if next_state is not None:
                target = reward + gamma * torch.max(agent.target_model(torch.tensor(next_state)) , dim= 1)[0]
            else:
                target = torch.tensor(reward)    
                
                
#             qqvalues= agent.model(torch.tensor(state))
#             q_value, mm = torch.max(qqvalues, dim = 1) 
#             loss = ((torch.abs(target - q_value))**0.5).sum()   
                
#             q_values = self.model(torch.tensor(state))

#             selected_rows = q_values[torch.arange(q_values.size(0)), action]
    
#             error = torch.abs(selected_rows - target)
#             darkexperienceloss= ((torch.abs(logits-qqvalues))**2)
#             darkexperienceloss= torch.sum(darkexperienceloss, dim= 1)
#             dd= darkexperienceloss.sum() 
            
#             loss = loss+ 0.2 * darkexperienceloss
            qqvalues = agent.model(torch.tensor(state))
            q_value, _ = torch.max(qqvalues, dim=1)
            loss3 = ((torch.abs(target - q_value)) ** 2).sum()
            
            darkexperienceloss = ((torch.abs(logits - qqvalues)) ** 2).sum()
            loss2 = 0.2 * darkexperienceloss 
            loss = loss3 + loss2  # Add the two losses
            loss = loss3  # Compute the sum of the combined losses

           
            #print(loss)# Adding to the loss
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
# Backward pass
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)  # Set retain_graph=True to retain the computational graph
            self.optimizer.step()

#         priorities = [abs(sample[5]) + 1e-6 for sample in samples]
#         self.memory.update_priorities(range(len(samples)), priorities)

#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay

    def CopyCurrent_and_targetDQNS(self):
        self.target_model.load_state_dict(self.model.state_dict())

def reinforcement_train(net, train_loader):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
    net.train()

    agent = DoubleDQNAgent(1, 2, 10000) 
    for epoch in range(args.epochs):
        k=0
        memory = []
        probability_weights = []         
        for features, labels in train_loader:
            feature_arrays= features
            random.shuffle(feature_arrays)
            agent = DoubleDQNAgent(1, 2, 10000)
            k=k+1
            if k % 5 == 0:
               agent.CopyCurrent_and_targetDQNS()
            labels = labels.type(torch.LongTensor)
            optimizer.zero_grad()
            outputs = net(features)
            targets = get_targets(outputs, labels)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            for feature, label in zip(features, labels):
                memory.append((feature, label))
                
            probability_weights.extend(get_probability_weights(outputs, labels))
        
        # sample = get_random_sample(memory)
        sample = get_prioritized_sample(memory, probability_weights)
        
        # Train on the sampled data - Experience replay
        for epoch in range(args.epochs):
            for features, labels in sample:
                optimizer.zero_grad()
                outputs = net(features)
                targets = get_targets(outputs, labels)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

def dueling_reinforcement_train(net, train_loader):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
    net.train()

    agent = DuelingDDQNAgent(1, 2, 10000) 
    for epoch in range(args.epochs):
        k=0
        memory = []
        probability_weights = []         
        for features, labels in train_loader:
            
            feature_arrays= features
            random.shuffle(feature_arrays)
            agent = DuelingDDQNAgent(1, 2, 10000)
            k=k+1
            if k % 5 == 0:
               agent.CopyCurrent_and_targetDQNS()
            labels = labels.type(torch.LongTensor)
            optimizer.zero_grad()
            outputs = net(features)
            targets = get_targets(outputs, labels)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            for feature, label in zip(features, labels):
                memory.append((feature, label))
                
            probability_weights.extend(get_probability_weights(outputs, labels))
        
        # sample = get_random_sample(memory)
        sample = get_prioritized_sample(memory, probability_weights)
        
        # Train on the sampled data - Experience replay
        for epoch in range(args.epochs):
            for features, labels in sample:
                optimizer.zero_grad()
                outputs = net(features)
                targets = get_targets(outputs, labels)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()          

def get_targets(outputs, labels):
    targets = []
    for output, label in zip(outputs, labels):
        
        output = output.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        prediction = np.argmax(array_softmax(output))
        reward = get_reward(prediction, label)
        
        if(reward == 1):
            target = prediction
        else:
            target = label
        
        targets.append(target)
        
    return torch.from_numpy(np.asarray(targets)).float().type(torch.LongTensor)


def get_reward(prediction, label):
    if(prediction == label):
        return 1
    else:
        return 0


def get_random_sample(memory):
    length = len(memory)
    sample = random.sample(memory, length//10)
    
    features = []
    labels = []
    for x, y in sample:
        features.append(x.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
        
    return get_tensor_loader(features, labels)
        

def get_prioritized_sample(memory, weights):
    weights = normalize(weights)
    length = len(memory)
    indices = np.arange(length)
    
    sample_indices = np.random.choice(indices, size=length//10, replace=False, p=weights)
    sample = list(itemgetter(*sample_indices)(memory))
    
    features = []
    labels = []
    for x, y in sample:
        features.append(x.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
        
    return get_tensor_loader(features, labels)


def array_softmax(arr):
    return np.exp(arr) / np.sum(np.exp(arr), axis=0)


def get_probability_weights(outputs, labels):
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    weights = []
    for output_vector, label in zip(outputs, labels):
        target_vector = get_target_vector(label)
        error_vector = [abs(i-j) for i, j in zip(output_vector, target_vector)]
        error = sum(error_vector)
        weights.append(pow(error, args.per_exponent))
        
    return weights


def get_target_vector(label):
    if(label == 0):
        return [1, 0]
    
    if(label == 1):
        return [0, 1]
    
    
def normalize(weights):
    total_sum = sum(weights)
    result = [x/total_sum for x in weights]
    return result