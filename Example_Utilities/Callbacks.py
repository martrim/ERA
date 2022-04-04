from datetime import datetime
from math import isnan
from tensorflow.keras.callbacks import Callback, EarlyStopping, LearningRateScheduler


def lr_scheduler(epoch):
    lr = 1e-4
    if epoch > 60:
        lr *= 1e-6
    elif epoch > 50:
        lr *= 1e-5
    elif epoch > 40:
        lr *= 1e-4
    elif epoch > 30:
        lr *= 1e-3
    elif epoch > 20:
        lr *= 1e-2
    elif epoch > 10:
        lr *= 1e-1
    return lr


class StopIfNaN(Callback):
    def __init__(self):
        super(Callback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if isnan(logs['val_loss']):
            print(f'Stopping training: NaN value after epoch {epoch}.')
            self.model.stop_training = True


class PrintTime(Callback):
    def __init__(self):
        super(PrintTime, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        now = datetime.now()
        print(f'Starting epoch {epoch}: {now.strftime("%H:%M:%S | %d/%m/%Y")}')

    def on_epoch_end(self, epoch, logs=None):
        now = datetime.now()
        print(f'Ending epoch {epoch}: {now.strftime("%H:%M:%S | %d/%m/%Y")}')


def get_callbacks():
    callbacks = [PrintTime(), StopIfNaN()]
    # Early Stopping
    if False:
        callbacks.append(EarlyStopping(monitor='val_accuracy', min_delta=0, patience=3, verbose=0,
                                       mode='auto', baseline=None))
    callbacks.append(LearningRateScheduler(lr_scheduler))
    return callbacks
