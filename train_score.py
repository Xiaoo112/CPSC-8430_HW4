import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Define the filenames and labels for the plots
    files = [
        ('inception_score_dcgan.csv', 'DCGAN', '^-', 'blue'),
        ('inception_score_wgan.csv', 'WGAN', 'o-', 'green'),
        ('inception_score_acgan.csv', 'ACGAN', 's-', 'red')
    ]
    data = []

    # Plot styling
    plt.figure(figsize=(10, 5))  # Set the figure size
    plt.style.use('ggplot')  # Use ggplot style for a change
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')  # Customize grid lines

    # Load the data from each file
    for filename, label, marker, color in files:
        df = pd.read_csv(filename)
        df.columns = df.columns.str.strip()
        data.append(df)
        plt.plot(df['epoch'], df['inception_score'], marker, label=label, linewidth=2, color=color)

    # Customize fonts and labels
    plt.xlabel("Epoch", fontsize=14, fontweight='bold')
    plt.ylabel("Score", fontsize=14, fontweight='bold')
    plt.title("Scores of GAN Variants Over Epochs", fontsize=16, fontweight='bold')
    plt.legend()

    # Saving and showing the plot
    plt.savefig("scores.png")
    plt.close()
    print("Image scores.png exported!")


if __name__ == "__main__":
    main()