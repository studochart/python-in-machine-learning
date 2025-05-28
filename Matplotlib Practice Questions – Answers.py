import matplotlib.pyplot as plt

# 1. Simple Line Plot
x = [0, 1, 2, 3, 4]
y = [0, 1, 4, 9, 16]
plt.plot(x, y, color='red')
plt.title("Simple Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# 2. Multiple Lines in One Plot
x = [1, 2, 3]
y1 = [1, 4, 9]
y2 = [2, 3, 5]
plt.plot(x, y1, label="y = x^2", color="blue")
plt.plot(x, y2, label="y = 2x + 1", color="green")
plt.title("Multiple Lines")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# 3. Histogram of Random Data
data = [5, 7, 7, 8, 9, 10, 10, 10, 11, 12]
plt.hist(data, bins=4, color='lightblue')
plt.title("Sample Histogram")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()

# 4. Bar Plot of Favorite Fruits
fruits = ["Apple", "Banana", "Mango", "Orange"]
values = [15, 25, 10, 20]
plt.bar(fruits, values, color=['red', 'yellow', 'orange', 'green'])
plt.title("Favorite Fruits")
plt.xlabel("Fruit")
plt.ylabel("Votes")
plt.show()

# 5. Custom Line Styles
x = [1, 2, 3, 4, 5]
y = [i ** 2 for i in x]
plt.plot(x, y, linestyle='--', marker='o', color='purple')
plt.title("y = x^2 with Dashed Line and Markers")
plt.grid(True)
plt.show()

# 6. Grid Lines and Styling
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
plt.plot(x, y, color='orange', linewidth=2)
plt.title("Line Plot with Grid")
plt.grid(True)
plt.show()

# 7. Vertical vs Horizontal Bar Chart
students = ["Alice", "Bob", "Charlie", "David"]
marks = [85, 90, 78, 88]
# Vertical
plt.bar(students, marks, color='skyblue')
plt.title("Student Marks (Vertical)")
plt.ylabel("Marks")
plt.show()
# Horizontal
plt.barh(students, marks, color='lightgreen')
plt.title("Student Marks (Horizontal)")
plt.xlabel("Marks")
plt.show()

# 8. Pie Chart of Activities
activities = ['Sleep', 'Study', 'Play', 'Eat']
hours = [8, 6, 4, 6]
explode = (0.1, 0, 0, 0)
plt.pie(hours, labels=activities, autopct='%1.1f%%', explode=explode, startangle=90)
plt.title("Daily Routine")
plt.show()

# 9. Bar Plot with Edge Color
categories = ["A", "B", "C"]
values = [10, 20, 15]
plt.bar(categories, values, color='green', edgecolor='black')
plt.title("Bar Plot with Black Edges")
plt.show()

# 10. Line Plot with Annotations
x = [1, 2, 3, 4]
y = [2, 4, 6, 8]
plt.plot(x, y, marker='o')
plt.title("Annotated Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.text(2, 4, "Here!", fontsize=12, color='red')
plt.grid(True)
plt.show()
