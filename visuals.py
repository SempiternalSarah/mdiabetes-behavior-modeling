import matplotlib.pyplot as plt

def make_param_title(title, report):
    model = report["model_kw"]
    train = report["train_kw"]
    epochs = train["epochs"]
    train_str = f"E{epochs}"
    lfn = model["lossfn"]
    lr = model["opt_kw"]["lr"]
    hid = model["hidden_size"]
    data = report['params']['data_kw']
    data_str = f"Resp{data['top_respond_perc']}States{int(data['include_state'])}Expanded{int(data['expanded_states'])}"
    model_str = f"{lfn}LR={lr},"
    return f"{title}\n{train_str}\n{report['params']['model_name']}{model_str}\n{data_str}"

def title_to_fname(title):
    return "img/"+title.replace(" ", "").replace("\n", "-")+".png"

def subplots(nrows=1, ncols=1, **kw):
    fig, ax = plt.subplots(
        nrows=nrows, ncols=ncols, 
        edgecolor="w", facecolor="w", **kw)
    return fig, ax
                 
class Plotter:

    def response_evaluations(report):
        res = report["results"]
        fig, ax = subplots(nrows=len(res))
        for i, r in enumerate(res):
            foo = ax[i].imshow(r, cmap="twilight", vmin=-3, vmax=3)
        ax[len(ax)//2].set_ylabel("Question #")
        ax[-1].set_xlabel("Response Week #")
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        plt.colorbar(foo, cax=cbar_ax)
        title = make_param_title(
            "Error in Predicted Responses per User by Week",
            report
        )
        ax[0].set_title(title, y=1.05)
        fig.savefig(title_to_fname(title))
    
    def training_loss(report, direct):
        loss = report["loss"]
        fig, ax = subplots()
        xes = [report['train_kw']['rec_every'] * x for x in range(loss.shape[0])]
        for i in range(loss.shape[1]):
            curve = loss[:,i]
            ax.plot(xes, curve)
        title = make_param_title(
            "UserLoss",
            report
        )
        ax.set_title("Training Loss of Selected Users", y=1.05)
        ax.set_ylabel("Error")
        ax.set_xlabel("Training epoch")
        fig.savefig(f'{direct}{title_to_fname(title)}')
