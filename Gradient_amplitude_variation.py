import numpy as np
import matplotlib.pyplot as plt

def Gradient_amplitude_variation(amplitude_of_loss_gradient, test_loss):
    plt.rc('font',family='Times New Roman')



    with_flooding = np.random.choice([True, False], size=len(epochs))
    plt.figure(figsize=(8, 6))
    plt.scatter(amplitude_of_loss_gradient[with_flooding], test_loss[with_flooding],
                c=epochs[with_flooding], cmap='summer', s=100, marker='o', label='method1')
    plt.scatter(amplitude_of_loss_gradient[~with_flooding], test_loss[~with_flooding],
                c=epochs[~with_flooding], cmap='summer', s=100, marker='+', label='method2')

    plt.colorbar(label='Epoch')
    plt.xlabel('Amplitude of Test Loss Gradient')
    plt.ylabel('Test Loss')
    plt.legend()
    plt.title('Test Loss vs. Amplitude of Test Loss Gradient')
    save_path = './Amplitude_of_Test_Loss_Gradient.png'
    plt.savefig(save_path)
    plt.show()
    plt.close()

if __name__ == '__main__':
    np.random.seed(0)
    epochs = np.random.randint(1, 30, 100)
    # print(epochs)
    amplitude_of_loss_gradient = np.random.rand(100)
    test_loss = np.random.rand(100) * 1.5
    Gradient_amplitude_variation(amplitude_of_loss_gradient,test_loss)
