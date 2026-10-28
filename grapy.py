plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.plot(x_test_original[order] , y_true[order],label=' actual  value ')
plt.plot(x_test_original[order] , y_pred[order] , label ='predicted values ')
plt.title("actual vs predicted value")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)



plt.subplot(1,3,2)

plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'] ,label='validation loss')
plt.title('training vs validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.grid(True)

plt.subplot(1,3,3)
plt.plot(history.history['mae'], label='training loss')
plt.plot(history.history['val_mae'] ,label='validation loss')
plt.title('training vs validation mae')
plt.xlabel('epochs')
plt.ylabel('mae')
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()
