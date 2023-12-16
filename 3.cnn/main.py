import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras import layers, models, losses, optimizers, datasets, utils

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images.shape, train_labels.shape, test_images.shape, test_labels.shape

np.set_printoptions(linewidth=200, precision=2)

unique, counts = np.unique(train_labels, return_counts=True)
num_labels = len(unique)
f"Train labels: {dict(zip(unique, counts))}"

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def save_trained_model():
    global train_images, test_images

    plt.figure(figsize=(10, 10))
    for idx, idx_img in enumerate(np.random.randint(0, 4999, 25)):
        plt.subplot(5, 5, idx + 1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(train_images[idx_img])
        plt.xlabel(class_names[train_labels[idx_img][0]])
    plt.show()

    train_images, test_images = train_images / 255, test_images / 255

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu')
    ])

    model.summary()

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.compile(
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizers.Adam(),
        metrics=["accuracy"],
    )

    history = model.fit(
        train_images, train_labels,
        validation_data=(test_images, test_labels),
        batch_size=256,
        epochs=10
    )

    test_scores = model.evaluate(test_images, test_labels, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])

    model.save("mnist.h5")
    del model


if __name__ == "__main__":
    # epochs= 10으로 학습한 모델을 로드한다.
    # epochs를 증가시켜 학습시킨 모델은 더욱 정확한 예측 결과를 생성한다.
    model = models.load_model("mnist.h5")

    test_scores = model.evaluate(test_images, test_labels, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])

    plt.figure(figsize=(10, 10))
    for idx, idx_img in enumerate(np.random.randint(0, 4999, 25)):
        output = model.predict(test_images[idx_img].reshape(1, 32, 32, 3))
        plt.subplot(5, 5, idx + 1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(test_images[idx_img].reshape(32, 32, 3))
        plt.xlabel(class_names[np.argmax(output)])
    plt.show()


