import numpy as np
import random
import matplotlib.pyplot as plt
import csv

def read_dataset(filepath):
    dataset = []
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            attributes = list(map(float, row[2:]))  # Use features from column 2 onward
            label = 1 if row[1] == 'M' else 0  # Malignant = 1, Benign = 0
            dataset.append((attributes, label))
    return dataset

def standardize(data):
    features = np.array([sample[0] for sample in data])
    labels = np.array([sample[1] for sample in data])
    normalized_features = (features - features.mean(axis=0)) / features.std(axis=0)
    return normalized_features, labels

def create_folds(data, num_folds=10):
    random.shuffle(data)
    fold_length = len(data) // num_folds
    return [data[i * fold_length:(i + 1) * fold_length] for i in range(num_folds)]



class MLP:
    def __init__(self, input_dim, hidden_dims, output_dim):
        self.architecture = [input_dim] + hidden_dims + [output_dim]
        self.weights = [np.random.randn(self.architecture[i], self.architecture[i+1]) * 0.1 
                        for i in range(len(self.architecture) - 1)]
        self.biases = [np.random.randn(1, self.architecture[i+1]) * 0.1 
                       for i in range(len(self.architecture) - 1)]

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.activations = [X]
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.activations.append(self.sigmoid(z))
        z_out = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        return MLP.sigmoid(z_out)

    def classify(self, X):
        predictions = self.forward(X)
        return np.argmax(predictions, axis=1)

class GA(MLP):
    def __init__(self, pop_size, mutation_prob, max_generations, input_size, hidden_layers, output_size, 
                 tournament_size=3, elite_fraction=0.025, learning_rate=0.01):
        super().__init__(input_size, hidden_layers, output_size)
        self.pop_size = pop_size
        self.mutation_prob = mutation_prob
        self.max_generations = max_generations
        self.tournament_size = tournament_size
        self.elite_count = int(pop_size * elite_fraction)
        self.learning_rate = learning_rate

    def initialize_population(self):
        return [MLP(self.architecture[0], self.architecture[1:-1], self.architecture[-1]) 
                for _ in range(self.pop_size)]

    def crossover(self, parent1, parent2):
        offspring = MLP(parent1.architecture[0], parent1.architecture[1:-1], parent1.architecture[-1])
        for i in range(len(parent1.weights)):
            mask = np.random.rand(*parent1.weights[i].shape) > 0.5
            offspring.weights[i] = np.where(mask, parent1.weights[i], parent2.weights[i])
        return offspring

    def mutate(self, model):
        for i in range(len(model.weights)):
            if np.random.rand() < self.mutation_prob:
                model.weights[i] += np.random.randn(*model.weights[i].shape) * self.learning_rate

    def evaluate(self, model, X, y):
        predictions = model.classify(X)
        return np.mean(predictions == y)

    def tournament_selection(self, population, X, y):
        tournament = random.sample(population, self.tournament_size)
        tournament = sorted(tournament, key=lambda nn: self.evaluate(nn, X, y), reverse=True)
        return tournament[0]

    def optimize(self, X_train, y_train, X_val, y_val):
        population = self.initialize_population()
        for gen in range(self.max_generations):
            population = sorted(population, key=lambda nn: self.evaluate(nn, X_val, y_val), reverse=True)
            new_population = population[:self.elite_count]

            while len(new_population) < self.pop_size:
                parent1 = self.tournament_selection(population, X_val, y_val)
                parent2 = self.tournament_selection(population, X_val, y_val)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)
            
            population = new_population 

        best_model = population[0]
        return best_model

def calculate_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    sensitivity = tp / (tp + fn) if tp + fn > 0 else 0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0
    precision = tp / (tp + fp) if tp + fp > 0 else 0

    return tp, tn, fp, fn, sensitivity, specificity, precision

def plot_confusion_matrix(tp, tn, fp, fn):
    cm = np.array([[tn, fp], [fn, tp]])
    labels = ['B', 'M']  

    plt.matshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')

    plt.xticks([0, 1], labels)  
    plt.yticks([0, 1], labels) 
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (B=Benign, M=Malignant)')
    plt.show()

# K-Fold cross-validation to evaluate the model
def cross_validate(data, folds=10):
    partitions = create_folds(data, folds)
    fold_scores = []

    for i in range(folds):
        val_data = partitions[i]
        train_data = [sample for j in range(folds) if j != i for sample in partitions[j]]
        
        X_train, y_train = standardize(train_data)
        X_val, y_val = standardize(val_data)

        optimizer = GA(
            pop_size=20, mutation_prob=0.01, max_generations=100, 
            input_size=X_train.shape[1], hidden_layers=[20, 10], output_size=2
        )

        best_network = optimizer.optimize(X_train, y_train, X_val, y_val)
        score = optimizer.evaluate(best_network, X_val, y_val)
        fold_scores.append(score)
        
        y_pred = best_network.classify(X_val)
        tp, tn, fp, fn, sensitivity, specificity, precision = calculate_metrics(y_val, y_pred)
        
        print(f"Fold {i + 1} Accuracy: {score:.4f}")
        print(f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, Precision: {precision:.4f}")

    avg_score = np.mean(fold_scores)
    print(f"Average Accuracy: {avg_score:.4f}")
    return fold_scores, avg_score, optimizer, best_network

if __name__ == '__main__':

    filepath = 'dataset.csv'
    dataset = read_dataset(filepath)
    
    fold_scores, avg_score, optimizer, best_model = cross_validate(dataset)

    X, y = standardize(dataset)
    y_pred = best_model.classify(X)
    
    tp, tn, fp, fn, sensitivity, specificity, precision = calculate_metrics(y, y_pred)
    print(f"Overall Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, Precision: {precision:.4f}")
    
    plot_confusion_matrix(tp, tn, fp, fn)
