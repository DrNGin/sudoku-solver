import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np

# تنظیم بذر تصادفی برای تکرارپذیری
tf.random.set_seed(42)
np.random.seed(42)

# 1. بارگذاری و پیش‌پردازش دیتاست MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# نرمال‌سازی تصاویر (تبدیل به مقادیر 0 تا 1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# تغییر شکل تصاویر برای ورودی مدل (اضافه کردن کانال)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 2. افزودن نمونه‌های خالی به دیتاست
# تصاویر خالی (سیاه) به‌عنوان کلاس 0
num_empty_samples = 10000  # تعداد نمونه‌های خالی
empty_images = np.zeros((num_empty_samples, 28, 28, 1), dtype='float32')
empty_labels = np.zeros(num_empty_samples, dtype='int')

# ترکیب نمونه‌های خالی با دیتاست اصلی
x_train = np.concatenate([x_train, empty_images], axis=0)
y_train = np.concatenate([y_train, empty_labels], axis=0)

# تبدیل برچسب‌ها به فرمت one-hot
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 3. تعریف مدل پیشرفته CNN
model = models.Sequential([
    # لایه‌های کانولوشنی اولیه
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # لایه‌های کانولوشنی عمیق‌تر
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # لایه‌های کانولوشنی اضافی برای دقت بالاتر
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # لایه‌های کاملاً متصل
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # 10 کلاس (0 تا 9)
])

# 4. کامپایل مدل
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# نمایش معماری مدل
model.summary()

# 5. آموزش مدل
history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=20,
                    validation_data=(x_test, y_test),
                    shuffle=True)

# 6. ارزیابی مدل
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"دقت مدل روی دیتاست تست: {test_accuracy:.4f}")

# 7. ذخیره مدل
model.save('digit_models.h5')
print("مدل با نام 'digit_models.h5' ذخیره شد.")
