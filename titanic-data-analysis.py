import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import xlabel

titanic_data = pd.read_csv('titanic.csv')
print(titanic_data)
titanic_data.dropna(subset=['age', 'fare'], inplace=True)
plt.figure(figsize=(10,10))
plt.subplot(3,2,1)
clasa_3 = titanic_data[titanic_data["pclass"] == 3].shape[0]
clasa_2 = titanic_data[titanic_data["pclass"] == 2].shape[0]
clasa_1 = titanic_data[titanic_data["pclass"] == 1].shape[0]
labels1 = ["3rd Class", "2nd Class", "1st Class"]
plt.bar(0,clasa_3, label="3rd Class", color="lightgreen")
plt.bar(1, clasa_2, label = "2nd Class", color="orange")
plt.bar(2, clasa_1, label="1st Class", color="skyblue")
plt.xticks([0,1,2], ["3rd Class", "2nd Class", "1st Class"])
plt.text(0, clasa_3, clasa_3)
plt.text(1, clasa_2, clasa_2)
plt.text(2, clasa_1, clasa_1)
plt.xlabel("Classes")
plt.ylabel("Number of Passengers")
plt.legend()
plt.title("Distribution of classes on Titanic")
plt.subplot(3,2,2)
kids = int(titanic_data[titanic_data["age"] < 14].shape[0])
teenager = int(titanic_data[(titanic_data["age"] >= 14) & (titanic_data["age"] < 18)].shape[0])
adults = int(titanic_data[(titanic_data["age"] > 18) & (titanic_data["age"] <= 60)].shape[0])
old_people = int(titanic_data[titanic_data["age"] > 60].shape[0])
labels2 = ["Kids", "Teenagers", "Adults", "Old People"]
colors = ["lightblue", "lightgreen", "orange", "pink"]
plt.pie([kids, teenager, adults, old_people], labels=labels2, colors = colors, autopct="%1.1f%%", explode=(0,0,0.05,0))
plt.title("Distribution of Passengers by Age on Titanic")
plt.subplot(3,2,3)
survivors = titanic_data[titanic_data["survived"] == 1].shape[0]
deaths = titanic_data[titanic_data["survived"] == 0].shape[0]
labels = ["Survivors", "Deaths"]
plt.bar(0, survivors, label="Survivors", color="green")
plt.bar(1, deaths, label="Deaths", color="red")
plt.xticks([0,1], ["Survivors", "Deaths"])
plt.text(0, survivors, survivors)
plt.text(1, deaths, deaths)
plt.title("Survival / Deaths on Titanic")
plt.ylabel("Number of Passengers")
plt.legend()
plt.subplot(3,2,4)
fare = titanic_data["fare"]
plt.hist(fare, bins=30, color="blue")
plt.title("Distribution of Passenger Fares on the Titanic")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.subplot(3,2,5)
survivors_males = titanic_data[(titanic_data["survived"] == 1) & (titanic_data["sex"] == "male")].shape[0]
survivors_females = titanic_data[(titanic_data["survived"] == 1) & (titanic_data["sex"] == "female")].shape[0]
labels3 = ["Male Survivors", "Female Survivors"]
plt.bar(0, survivors_males, label="Male Survivors", color="#F54927")
plt.bar(1, survivors_females, label= "Female Survivors", color="#9C27F5")
plt.text(0, survivors_males, survivors_males)
plt.text(1, survivors_females, survivors_females)
plt.xticks([0,1], ["Male Survivors", "Female Survivors"])
plt.legend()
plt.title("Number of Survivors by Gender")
plt.subplot(3,2,6)
survival_rate_class1 = titanic_data[titanic_data["pclass"] == 1]["survived"].mean() * 100
survival_rate_class2 = titanic_data[titanic_data["pclass"] == 2]["survived"].mean() * 100
survival_rate_class3 = titanic_data[titanic_data["pclass"] == 3]["survived"].mean() * 100
rates = [survival_rate_class1, survival_rate_class2, survival_rate_class3]
labels = ["1st Class", "2nd Class", "3rd Class"]
plt.bar(labels, rates, color=["darkblue", "skyblue", "lightgreen"])
for i, rate in enumerate(rates):
    plt.text(i, rate + 1, f"{rate:.1f}%", ha="center")
plt.title("Survival Rate by Class")
plt.ylabel("Survival Rate (%)")
plt.tight_layout()
plt.savefig('plots-overview.png', dpi=300)
plt.show()



