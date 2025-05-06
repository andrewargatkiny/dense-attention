# Welcome to the Project!


This guide was generated partially by ChatGPT, however, it describes the exact 
workflow expected when contributing to this project.

---

## 🛠Contribution Workflow

### 1. Fork the Repository

- Go to the main repository page.
- Click the **Fork** button (top-right corner).
- This creates **your own copy** of the repository.

### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/REPO_NAME.git
cd REPO_NAME
```

### 3. Create a Feature Branch

Always branch from the latest version of `dev`.

```bash
git checkout dev
git pull origin
git checkout -b your-feature-name
```

> 🔄 Use descriptive branch names like:  
> `mamba_inference`, `sota-configs-lra`, or `weights-biases-support`.

---

## 💻 Make Your Changes

- Write clean, readable code.
- Don't be shy to write comments. They should ideally help a person unfamiliar with 
the task to easily understand the intricacies of your implementation.
- Test your code locally (if applicable).

---

## ✅ Commit Your Changes

```bash
git add .
git commit -m "Add: Feature description"
git push origin your-feature-name
```

---

## 📬 Create a Pull Request

- Go to your fork on GitHub.
- Click **Compare & pull request**.
- Base branch: `dev` on **upstream/original repo**
- Compare branch: your `feature/...` branch.
- Fill in title and description.
- Assign the pull request to the maintainer (e.g., @REPO_OWNER).
- Submit the PR.

---

## 🧹 After Your PR Is Merged

You can delete your branch after it’s merged:

```bash
git checkout dev
git pull upstream dev
git branch -d feature/your-feature-name
```


