"""
Evaluates all 6 datasets.
"""
import make_training_data.synthesise_training_data as make
import make_training_data.format_training_data as format
import utility.hyperparam_opt as opt
import patchy_san.cnn as cnn
import utility.error_metrics as error
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

train_acc = {}
test_acc = {}
train_error = {}
test_error = {}
cv = {}
pr_report = {}
pr = {}


def eval_datasets(graphs, dataset):
    xpn, xpe, xe, y = format.process_training_examples(graphs)
    inputs = [xpn, xpe, xe]
    model = cnn.build_model(0.005, "sigmoid")
    history = model.fit(inputs, y, epochs=20, batch_size=10, validation_split=0.2, shuffle=True)

    train_acc[dataset] = history.history['acc']
    test_acc[dataset] = history.history['val_acc']

    input20 = [inputs[0][1600:], inputs[1][1600:], inputs[2][1600:]]
    y20 = y[1600:]

    pr_report[dataset] = error.get_precision_recall(input20, y20, model)

    test_error[dataset] = error.get_error_bound(input20, y20, model)

    input80 = [inputs[0][:1600], inputs[1][:1600], inputs[2][:1600]]
    y80 = y[:1600]
    train_error[dataset] = error.get_error_bound(input80, y80, model)
    acc, loss = opt.cross_validation(inputs, y, 10, 20, 0.005, "sigmoid")
    cv[dataset] = acc

    prediction_probs = model.predict(inputs)
    pr[dataset] = metrics.precision_recall_curve(y.argmax(axis=1), prediction_probs[:, 0], 0)


training_graphs = make.get_graphs_test_negative_data_4_easy()
eval_datasets(training_graphs, 1)

training_graphs = make.get_graphs_test_negative_data_4()
eval_datasets(training_graphs, 2)

training_graphs = make.get_graphs_altered_cmdlines(10, True)
eval_datasets(training_graphs, 3)

training_graphs = make.get_graphs_altered_cmdlines(10)
eval_datasets(training_graphs, 4)

training_graphs = make.get_graphs_altered_cmdlines(20)
eval_datasets(training_graphs, 5)

training_graphs = make.get_graphs_altered_cmdlines(100)
eval_datasets(training_graphs, 6)

# CV accuracy
print("Cross validation accuracy.")
for i in range(1, 7):
    print("Accuracy for dataset " + str(i) + ": " + str(cv[i]))

print("Precision/Recall reports.")
for i in range(1, 7):
    print("PR for dataset " + str(i))
    print(pr_report[i])
    print("############")

# Plot precision recall curve
plt.figure(1)
plot_handles = []
for i in range(1, 7):
    plot_handles += plt.plot(pr[i][1], pr[i][0], label='Dataset ' + str(i))

plt.title("Precision recall curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(handles=plot_handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# Plot training accuracy
plt.figure(2)
plot_handles = []
for i in range(1, 7):
    plot_handles += plt.plot([t for t in range(1, 21)], train_acc[i], label='Dataset ' + str(i))

plt.title("Training accuracy against training epoch")
plt.xlabel("Training epoch")
plt.ylabel("Training accuracy")
plt.legend(handles=plot_handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# Plot test accuracy
plt.figure(3)
plot_handles = []
for i in range(1, 7):
    plot_handles += plt.plot([t for t in range(1, 21)], test_acc[i], label='Dataset ' + str(i))

plt.title("Test accuracy against training epoch")
plt.xlabel("Training epoch")
plt.ylabel("Test accuracy")
plt.legend(handles=plot_handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
