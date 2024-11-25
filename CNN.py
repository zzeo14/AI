# 61% accuracy
# Adam(lr = 1e-2)
# batch size = 128
max_epoch = 50

# Lists to store training and testing losses
tr_loss_saver = []
te_loss_saver = []

# Iterate over each epoch
for epoch in tqdm(range(max_epoch)):
    ### Train Phase

    # Initialize Loss and Accuracy for training phase
    train_loss = 0.0
    train_accu = 0.0

    # Iterate over the train_dataloader
    for idx, sample in enumerate(train_dataloader):
        # Call the train function to perform training on the current batch
        curr_loss, num_correct = train(model, optimizer, sample)

        # Update the total training loss and accuracy
        train_loss += curr_loss / len(train_dataloader)
        train_accu += num_correct / len(train_dataset)

    # Save the training loss
    tr_loss_saver.append(train_loss)

    # Save the model state after each epoch
    torch.save(model.state_dict(), 'recent.pth')

    ### Test Phase

    # Initialize Loss and Accuracy for testing phase
    test_loss = 0.0
    test_accu = 0.0

    # Iterate over the test_dataloader
    for idx, sample in enumerate(test_dataloader):
        # Call the test function to evaluate the model on the current batch
        curr_loss, num_correct = test(model, sample)

        # Update the total testing loss and accuracy
        test_loss += curr_loss / len(test_dataloader)
        test_accu += num_correct / len(test_dataset)

    # Save the testing loss
    te_loss_saver.append(test_loss)

    # Print the epoch-wise training and testing statistics
    print('[EPOCH {}] TR LOSS : {:.03f}, TE LOSS : {:.03f}, TR ACCU: {:.03f}, TE ACCU : {:.03f}'.format(epoch+1, train_loss, test_loss, train_accu, test_accu))
