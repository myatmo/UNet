
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_train_generator(train_df, batch_size=3, target_size=(256,256), seed=1, image_color_mode="grayscale"):
    # Data augmentation: define rotation angle, width and height shift, shear and zoom range, horizontal flip
    image_gen_args = dict(rotation_range=0.5,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='reflect',
                    rescale=1./255.
                    #validation_split=0.30
                    )

    mask_gen_args = dict(rotation_range=0.5,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='reflect',
                    #validation_split=0.30
                    )
    
    image_datagen = ImageDataGenerator(**image_gen_args)
    mask_datagen = ImageDataGenerator(**mask_gen_args)

    image_generator = image_datagen.flow_from_dataframe(dataframe = train_df,
                                                        x_col = "images",
                                                        y_col = None,
                                                        #subset = "training",
                                                        batch_size = batch_size,
                                                        seed = 1,
                                                        class_mode = None,
                                                        color_mode = image_color_mode,
                                                        target_size = target_size)


    mask_generator = mask_datagen.flow_from_dataframe(dataframe = train_df,
                                                        x_col = "masks",
                                                        y_col = None,
                                                        #subset = "validation",
                                                        batch_size = batch_size,
                                                        seed = 1,
                                                        class_mode = None,
                                                        color_mode = image_color_mode,
                                                        target_size = target_size)

    train_generator = zip(image_generator, mask_generator)
    train_step_size = image_generator.n // image_generator.batch_size

    return train_generator, train_step_size


def get_test_generator(test_df, batch_size=3, target_size=(256,256), seed=1, image_color_mode="grayscale"):
    # Define ImageDataGenerator for testing images
    test_gen_args = dict(rescale=1./255.)
    test_datagen = ImageDataGenerator(**test_gen_args)
    test_generator = test_datagen.flow_from_dataframe(dataframe = test_df,
                                                x_col = "images",
                                                y_col = None,
                                                batch_size = batch_size,
                                                seed = 1,
                                                class_mode = None,
                                                color_mode = image_color_mode,
                                                target_size = target_size,
                                                shuffle=False)

    test_step_size = test_generator.n // test_generator.batch_size

    return test_generator, test_step_size