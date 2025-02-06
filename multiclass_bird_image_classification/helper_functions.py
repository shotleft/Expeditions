#!/usr/bin/env python
# coding: utf-8

# In[1]:

from wordcloud import WordCloud
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
import numpy as np
import pandas as pd

def create_word_cloud(sentiment, freq_dict, colormap, ):

    cloud = WordCloud(background_color='white', colormap = colormap)
    
    # generate word cloud
    cloud.generate_from_frequencies(freq_dict)

    # show
    plt.imshow(cloud)
    plt.axis('off')
    plt.title(f'''Top 20 terms for {sentiment} reviews
    ''', fontsize = 14)
    plt.show()
    
def plot_history(history_dict):
    fig, ax = plt.subplots(1,2, figsize=(15, 4))
    epochs = range(1, len(history_dict['loss'])+1)
    ax[0].plot(epochs, history_dict['loss'], marker='o', markersize=5, linestyle='dotted', color = 'lightseagreen', label='Training loss')
    ax[0].plot(epochs, history_dict['val_loss'], marker='o', markersize=7, color = 'darkgreen', label='Validation loss')
    ax[0].set_title('Training and validation loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[1].plot(epochs, history_dict['acc'], marker='o', markersize=5, linestyle='dotted', color = 'lightseagreen', label='Training accuracy')
    ax[1].plot(epochs, history_dict['val_acc'], marker='o', markersize=7, color= 'darkgreen', label='Validation accuracy')
    ax[1].set_title('Training and validation accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    plt.show()
    
def plot_cfmatrix(cfmatrix):
    group_names = ['TN','FP','FN','TP']
    group_counts = ['{0:0.0f}'.format(value) for value in
                    cfmatrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in
                         cfmatrix.flatten()/np.sum(cfmatrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    plt.figure(figsize=(4, 3)) 
    sns.heatmap(cfmatrix, annot=labels, fmt='', cmap = 'RdYlGn')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion matrix')
    plt.show()

def plot_sklearn_roc_curve(y_real, y_proba):
    fpr, tpr, _ = roc_curve(y_real, y_proba)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    roc_display.figure_.set_size_inches(4, 4)
    plt.title('ROC_AUC curve')
    plt.plot([0, 1], [0, 1], color = 'g')    

def populate_summary(summary, model, history):
    '''Takes model name and key figures (best validation loss, best validation accuracy, best epoch)
    from a training run and adds them to the specified dataframe for tracking purposes.'''
    best_val_loss = round(max(history.history['val_loss']), 4)
    best_val_accuracy = round(max(history.history['val_acc']), 4)
    best_epoch_loss = np.argmin(history.history['val_loss']) + 1
    new_row = pd.Series({'model_name':model.name, 
                         'best_val_loss':best_val_loss, 
                         'best_val_accuracy':best_val_accuracy,
                         'best_epoch_loss':best_epoch_loss,
                         })
    summary = pd.concat([summary, new_row.to_frame().T], ignore_index=True)
    return summary    
    
# In[ ]:




