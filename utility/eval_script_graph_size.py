import make_training_data.fetch_training_data as fetch
import make_training_data.synthesise_training_data as make
import make_training_data.format_training_data as format
import patchy_san.parameters as params
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
EPOCHS = 10
DATASETS = 1


def eval_datasets(graphs, dataset):
    xpn, xpe, xe, y = format.process_training_examples(graphs)
    inputs = [xpn, xpe, xe]
    model = cnn.build_model(0.005, "sigmoid")
    history = model.fit(inputs, y, epochs=EPOCHS, batch_size=10, validation_split=0.2, shuffle=True)

    train_acc[dataset] = history.history['acc']
    test_acc[dataset] = history.history['val_acc']

    input20 = [inputs[0][1600:], inputs[1][1600:], inputs[2][1600:]]
    y20 = y[1600:]

    pr_report[dataset] = error.get_precision_recall(input20, y20, model)

    test_error[dataset] = error.get_error_bound(input20, y20, model)

    input80 = [inputs[0][:1600], inputs[1][:1600], inputs[2][:1600]]
    y80 = y[:1600]
    train_error[dataset] = error.get_error_bound(input80, y80, model)
    acc, loss = opt.cross_validation(inputs, y, 10, EPOCHS, 0.005, "sigmoid")
    cv[dataset] = acc

    prediction_probs = model.predict(input20)
    pr[dataset] = metrics.precision_recall_curve(y20.argmax(axis=1), prediction_probs[:, 0], 0)


results = fetch.get_train_8_nodes_general()
training_graphs = make.get_graphs_n_nodes(results)
eval_datasets(training_graphs, 1)

# CV accuracy
print("Cross validation accuracy.")
for i in range(1, DATASETS+1):
    print("Accuracy for dataset " + str(i) + ": " + str(cv[i]))

print("Precision/Recall reports.")
for i in range(1, DATASETS+1):
    print("PR for dataset " + str(i))
    print(pr_report[i])
    print("############")

# Plot precision recall curve
plt.figure(1)
plot_handles = []
for i in range(1, DATASETS+1):
    plot_handles += plt.plot(pr[i][1], pr[i][0], label='Dataset ' + str(i))

plt.title("Precision recall curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(handles=plot_handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# Plot training accuracy
plt.figure(2)
plot_handles = []
for i in range(1, DATASETS+1):
    plot_handles += plt.plot([t for t in range(1, EPOCHS+1)], train_acc[i], label='Dataset ' + str(i))

plt.title("Training accuracy against training epoch")
plt.xlabel("Training epoch")
plt.ylabel("Training accuracy")
plt.legend(handles=plot_handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# Plot test accuracy
plt.figure(3)
plot_handles = []
for i in range(1, DATASETS+1):
    plot_handles += plt.plot([t for t in range(1, EPOCHS+1)], test_acc[i], label='Dataset ' + str(i))

plt.title("Test accuracy against training epoch")
plt.xlabel("Training epoch")
plt.ylabel("Test accuracy")
plt.legend(handles=plot_handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
