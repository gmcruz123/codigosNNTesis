from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import tensorflow as tf #Import tensor flow

#import iris_data
import fingers_data # File with data for NN training


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    #(train_x, train_y), (test_x, test_y) = iris_data.load_data()
    (train_x, train_y), (test_x, test_y) = fingers_data.load_data()


    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        print(key)
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier( #Dnn classifier , its backpropagation
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # The model must choose between 22 classes, alphabet letters.
        n_classes= 22,
        #Define optimizer
        optimizer = tf.train.ProximalAdagradOptimizer(
        learning_rate=0.07,
        l1_regularization_strength=0.001)
    )
    # Train the Model.
    classifier.train(
        input_fn=lambda:fingers_data.train_input_fn(train_x, train_y,
                                                 args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    #eval_result = classifier.evaluate(
    #    input_fn=lambda:fingers_data.eval_input_fn(test_x, test_y,
    #                                            args.batch_size))

    #print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = ['z']
    predict_x = {
        'menique': [71592],#Target from thumb
        'medio': [6503],#Target from index finger
        'indice':[26039],#Target from ring finger
        'pulgar': [23222],#Target from pinkie

    }

    predictions = classifier.predict(
        input_fn=lambda:fingers_data.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=args.batch_size))

    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

    for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        #print(template.format(iris_data.SPECIES[class_id],
        print(template.format(fingers_data.CLASSES[class_id],
                              100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)