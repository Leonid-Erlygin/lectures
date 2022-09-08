import tensorflow as tf
import numpy as np

import matplotlib as mpl
import cv2
import IPython.display as display
import PIL.Image

from tensorflow.keras.preprocessing import image


class DeepDream(tf.Module):
    def __init__(self, model, loss_func):
        self.model = model
        self.loss_func = loss_func

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.float32),
        )
    )
    def __call__(self, img, steps, step_size):
        print("Tracing")
        loss = tf.constant(0.0)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                # This needs gradients relative to `img`
                # `GradientTape` only watches `tf.Variable`s by default
                tape.watch(img)
                loss = self.loss_func(img, self.model)

                # Calculate the gradient of the loss with respect to the pixels of the input image.
            gradients = tape.gradient(loss, img)
            # Normalize the gradients.
            gradients /= tf.math.reduce_std(gradients) + 1e-8

            # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
            # You can update the image by directly adding the gradients (because they're the same shape!)
            img = img + gradients * step_size
            img = tf.clip_by_value(img, -1, 1)

        return loss, img


class DeepResp:
    def __init__(self, model, loss_func, steps, step_size):
        self.steps = steps
        self.step_size = step_size
        self.deepdream = DeepDream(model, loss_func)

    def deprocess(self, img):
        img = 255 * (img)
        return tf.cast(img, tf.uint8)

    # Display an image
    def show(self, img):
        display.display(PIL.Image.fromarray(cv2.resize(np.array(img), (256, 256))))

    def run_deep_dream_simple(self, img):
        # Convert from uint8 to the range expected by the model.
        step_size = tf.convert_to_tensor(self.step_size)
        steps_remaining = self.steps
        step = 0
        while steps_remaining:
            if steps_remaining > 100:
                run_steps = tf.constant(100)
            else:
                run_steps = tf.constant(steps_remaining)
            steps_remaining -= run_steps
            step += run_steps

            loss, img = self.deepdream(img, run_steps, tf.constant(step_size))
            display.clear_output(wait=True)
            self.show(self.deprocess(img))
            print("Step {}, loss {}".format(step, loss))

        result = self.deprocess(img)
        display.clear_output(wait=True)
        self.show(result)

        return result

    def __call__(self, image):
        return self.run_deep_dream_simple(image)
