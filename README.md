# Covariate-Shift-in-Machine-Learning

Paper published in IEEE Journal:

S. V. Chapaneri and D. J. Jayaswal, "Covariate Shift Adaptation for Structured Regression With Frank–Wolfe Algorithms," in IEEE Access, vol. 7, pp. 73804-73818, 2019, doi: 10.1109/ACCESS.2019.2920486.

https://ieeexplore.ieee.org/document/8727878

Book Chapter published in Handbook of Research on Machine Learning:
https://www.appleacademicpress.com/handbook-of-research-on-machine-learning-foundations-and-applications/9781774638682
 
![image](https://user-images.githubusercontent.com/17112412/208481417-f7a4def6-a93d-4c2a-a25f-86506bec425b.png)

A common assumption in most machine learning methods is that the training and test data samples belong to the same probability distribution. Given the joint training distribution $p_{\text{tr}}(\mathbf{x},\mathbf{y})$ and the joint test distribution $p_{\text{te}}(\mathbf{x},\mathbf{y})$ where $\mathbf{x} \in \mathcal{X}$ is the feature vector and $\mathbf{y} \in \mathcal{Y}$ is the target, the assumption is that the joint distributions remain the same, i.e. $p_{\text{tr}}(\mathbf{x},\mathbf{y}) = p_{\text{te}}(\mathbf{x},\mathbf{y})$.

However, this assumption is generally not valid for real-life test data. For example, consider the problem of predicting the price of an insurance policy based on age, income, employment type, etc. With the typical machine learning pipeline of pre-processing, data cleaning, feature selection and model training, the model performance can be evaluated on the test data. The test performance can degrade if the test distribution of age is different from the train distribution of age, i.e. if the model was learned on the age group of 15 to 45, but the test data also includes customers belonging to the age group of 50 and above. Due to this potential mismatch between the distributions, the test performance is always lower than the training performance.

There are broadly three types of dataset shift studied in the literature:

1. Covariate shift, which occurs due to a shift in the distribution of independent variables for $\mathcal{X} \to \mathcal{Y}$ problems. In this case, the conditional distributions remain the same but the marginal distributions of inputs differ, i.e. $p_{\text{tr}}(\mathbf{x},\mathbf{y}) = p_{\text{tr}}(\mathbf{y}|\mathbf{x})p_{\text{tr}}(\mathbf{x})$ and $p_{\text{te}}(\mathbf{x},\mathbf{y}) = p_{\text{te}}(\mathbf{y}|\mathbf{x})p_{\text{te}}(\mathbf{x})$, so $p_{\text{tr}}(\mathbf{y}|\mathbf{x}) = p_{\text{te}}(\mathbf{y}|\mathbf{x})$ and $p_{\text{tr}}(\mathbf{x}) \neq p_{\text{te}}(\mathbf{x})$.
 
2. Prior probability shift, which occurs due to a shift in the distribution of dependent variables. This dataset shift scenario is applicable to $\mathcal{Y} \to \mathcal{X}$ problems where the target variable determines the covariate values. An example is the field of medical diagnosis where the disease label determines the symptoms. In this case, we have $p_{\text{tr}}(\mathbf{x}|\mathbf{y}) = p_{\text{te}}(\mathbf{x}|\mathbf{y})$ and $p_{\text{tr}}(\mathbf{y}) \neq p_{\text{te}}(\mathbf{y})$.

3. Concept shift, also refered to as concept drift, can occur in both $\mathcal{X} \to \mathcal{Y}$ and $\mathcal{Y} \to \mathcal{X}$ problems due to a changing relationship between the independent and dependent variables. In this case, we have $p_{\text{tr}}(\mathbf{y}|\mathbf{x}) \neq p_{\text{te}}(\mathbf{y}|\mathbf{x})$ and $p_{\text{tr}}(\mathbf{x}) = p_{\text{te}}(\mathbf{x})$ for $\mathcal{X} \to \mathcal{Y}$ problems and $p_{\text{tr}}(\mathbf{x}|\mathbf{y}) \neq p_{\text{te}}(\mathbf{x}|\mathbf{y})$ and $p_{\text{tr}}(\mathbf{y}) = p_{\text{te}}(\mathbf{y})$ for $\mathcal{Y} \to \mathcal{X}$ problems.

![image](https://user-images.githubusercontent.com/17112412/208481066-ed384d6d-1aae-4af4-b969-1a4b8fbe58e9.png)

![image](https://user-images.githubusercontent.com/17112412/208481332-22af39c3-41cb-41e4-9439-b0d316c269b9.png)

The covariate shift can be corrected by estimating the importance weight $w(\mathbf{x}) = \frac{p_{\text{te}}(\mathbf{x})}{p_{\text{tr}}(\mathbf{x})}$ from the training and test data distributions. But this is a hard problem to solve due to the curse of dimensionality and is also unreliable for high-dimensional input data (e.g. KLIEP). A feasible solution is to thus learn the importance weight directly from the given data without the need to estimate the train and test probability densities.

![image](https://user-images.githubusercontent.com/17112412/208483465-d02cd86b-ae42-4a35-9ed8-a5664dfbd5ce.png)

![image](https://user-images.githubusercontent.com/17112412/208483495-a21ff2a8-5c43-46ca-9d3a-87d99a7e0f3b.png)

![image](https://user-images.githubusercontent.com/17112412/208483565-b6e0bc9d-6d17-43e7-aca5-f3c0d4f2f4ba.png)

![image](https://user-images.githubusercontent.com/17112412/208483642-188eaddd-c860-4845-9f61-9444de5e601f.png)

**GM-KLIEP method**:

![image](https://user-images.githubusercontent.com/17112412/208483697-e30abca1-423b-4328-a8e0-65109389ab4a.png)



In this work, the covariate shift is corrected in an unsupervised manner using the \textit{projection-free} Frank-Wolfe (FW) optimization algorithm. The Frank-Wolfe optimization algorithm can efficiently solve constrained convex optimization problems using the linearization principle and can obtain sparser solutions.

![image](https://user-images.githubusercontent.com/17112412/208481737-78a8da05-082b-45a1-a7b4-da05b537ac4a.png)

![image](https://user-images.githubusercontent.com/17112412/208481800-73405ebe-9bb1-43a0-a9de-ecfd27d6e468.png)

![image](https://user-images.githubusercontent.com/17112412/208481629-196de020-c017-4b5f-8adb-8a82b8ccb9d2.png)

![image](https://user-images.githubusercontent.com/17112412/208481874-6826e28e-94fe-4170-a98c-680a4f6496b6.png)

![image](https://user-images.githubusercontent.com/17112412/208481918-0ce99110-2601-4932-98d3-5dfeb610cac2.png)

![image](https://user-images.githubusercontent.com/17112412/208484097-613e027c-56db-4911-9ad2-7bdaa089ad26.png)

![image](https://user-images.githubusercontent.com/17112412/208481959-82b66dd5-11ad-42cf-9346-be9decabb582.png)

![image](https://user-images.githubusercontent.com/17112412/208482000-e45f02d7-7b89-4f26-b959-6711d4c640a3.png)

![image](https://user-images.githubusercontent.com/17112412/208482052-a1adafb8-1f05-4100-800d-fe1b4faf96b6.png)

![image](https://user-images.githubusercontent.com/17112412/208482112-21f206d3-8014-4347-87a6-f32d8832b0a6.png)

![image](https://user-images.githubusercontent.com/17112412/208482163-d164814e-a67e-4a03-98b5-c61f81adb163.png)

![image](https://user-images.githubusercontent.com/17112412/208482210-0e3b1240-2881-4dae-bd2f-ddaa285adccf.png)

![image](https://user-images.githubusercontent.com/17112412/208482244-12ed41e7-6728-48a4-a473-749e669626ea.png)

![image](https://user-images.githubusercontent.com/17112412/208482300-eb788325-6fd0-4d2a-8cd0-65e40a517560.png)


