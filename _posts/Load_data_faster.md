---
title:  "⚡️Load the same csv file 10X faster and with 10X less memory.⚡️"
date:   2020-10-04
layout: single
classes: wide
author_profile: False
comments: true
header:
    overlay_image: "https://cdn-images-1.medium.com/max/10574/0*I8sDqvRwyH3u3J8f"
    overlay_excerpt_color: "#333"
    show_overlay_excerpt: false
    actions:
    - label: "GitHub"
      url: "https://gist.github.com/PrudhviVajja"
---

# ⚡️ Load the same CSV file 10X times faster and with 10X less memory⚡️

Pandas, Dask, MultiProcessing, Etc…

![Photo by [Cara Fuller](https://unsplash.com/@caraventurera?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) (**Fastest Mammal**)](https://cdn-images-1.medium.com/max/10574/0*I8sDqvRwyH3u3J8f)*Photo by [Cara Fuller](https://unsplash.com/@caraventurera?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) (**Fastest Mammal**)*

Even when we have 1TB of Disk Storage, 8GB/16GB of RAM still pandas and much other data loading API struggles to load a 2GB file.

This is because when a process requests for memory, memory is allocated in two ways:

1. Contiguous Memory Allocation (consecutive blocks are assigned)

1. NonContiguous Memory Allocation(separate blocks at different locations)

Pandas use Contiguous Memory to load data into RAM because read and write operations are must faster on RAM than Disk(or SSDs).

* Reading from SSDs: ~16,000 nanoseconds

* Reading from RAM: ~100 nanoseconds

Before going into multiprocessing & GPU’s, etc… let us see how to use *pd.read_csv()* effectively.
> #### Pandas is fine for loading data and preprocessing but to train your models start using DataLoader from [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) or [PyTorch](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) or where ever you run your model.

![Photo by [Sebastian Yepes](https://unsplash.com/@sebasluna?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) (**fastest dog breed**)](https://cdn-images-1.medium.com/max/8000/0*k3JnCz15uVLILmMY)*Photo by [Sebastian Yepes](https://unsplash.com/@sebasluna?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) (**fastest dog breed**)*
> ### Note: If you are reading in your mobile you may not be able to scroll through the code. (Open the Gist for better readability.)

### ***1. use cols:***

Rather than loading data and removing unnecessary columns that aren’t useful when processing your data. load only the useful columns.

<script src="https://gist.github.com/PrudhviVajja/b2a87550ccc9ad1e6eb94df82339b2ff.js"></script>

### **2*.* Using correct dtypes for numerical data:**

Every column has it’s own dtype in a pandas DataFrame, for example, integers have int64, int32, int16 etc…

* int8 can store integers from -128 to 127.

* int16 can store integers from -32768 to 32767.

* int64 can store integers from -9223372036854775808 to 9223372036854775807.

Pandas assign int64 to integer datatype by default, therefore by defining correct dtypes we can reduce memory usage significantly.

<script src="https://gist.github.com/PrudhviVajja/5d0d90e2e0d6adaea50cc7d4acf58e22.js"></script>

**🔥 Pro Tip:** Use converters to replace missing values or NANs while loading data, especially for the columns that have predefined datatypes using dtype.

![](https://cdn-images-1.medium.com/max/2000/1*kwUv9LSh0L-XDmj_Znu6YA.png)

<iframe src="https://giphy.com/embed/l0O5Bb9UngW3Jo7vO" width="480" height="270" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/animation-dog-animated-l0O5Bb9UngW3Jo7vO">via GIPHY</a></p>

### ***3. Using correct dtypes for categorical columns:***

In my Dataset, I have a column Thumb which is by default parsed as a string, but it contains only a fixed number of values that remain unchanged for any dataset.

![](https://cdn-images-1.medium.com/max/2000/1*9nZXro8upEt_2cxPepCRKw.png)

And also columns such as Gender, etc.. can be stored as [categorical](https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html) values which reduces the memory from ~1000 KB to ~100 KB. (check the sats)

<script src="https://gist.github.com/PrudhviVajja/f76c4c0318cd3785a27618a09f53fad3.js"></script>

**🔥 Pro Tip: **If your DataFrame contains lots of empty values or missing values or NANs you can reduce their memory footprint by converting them to [Sparse Series](https://pandas.pydata.org/pandas-docs/stable/user_guide/sparse.html).

<script src="https://gist.github.com/PrudhviVajja/65b3c06c36b558f5de7af8fec49381fe.js"></script>

### ***4. nrows, skip rows***

Even before loading all the data into your RAM, it is always a good practice to test your functions and workflows using a small dataset and pandas have made it easier to choose precisely the number of rows (you can even skip the rows that you do not need.)

In most of the cases for testing purpose, you don’t need to load all the data when a sample can do just fine.

nrows The number of rows to read from the file.

```python
>>> Import pandas as pd
>>> df = pd.read_csv("train.csv", nrows=1000)
>>>len(df)
1000
```

skiprows Line numbers to skip (0-indexed) or the number of lines to skip (int) at the start of the file.

```python
# Can be either list or first N rows.
df = pd.read_csv('train.csv', skiprows=[0,2,5]) 
# It might remove headings
```

**🔥 Pro-Tip: **An Effective use of nrows is when you have more than 100’s of columns to check and define proper dtypes for each and every column. All of this overhead can be reduced using nrows as shown below.

```python
sample = pd.read_csv("train.csv", nrows=100) # Load Sample data

dtypes = sample.dtypes # Get the dtypes
cols = sample.columns # Get the columns

dtype_dictionary = {} 
for c in cols:
    """
    Write your own dtypes using 
    # rule 2
    # rule 3 
    """
    if str(dtypes[c]) == 'int64':
        dtype_dictionary[c] = 'float32' # Handle NANs in int columns
    else:
        dtype_dictionary[c] = str(dtypes[c])

# Load Data with increased speed and reduced memory.
df = pd.read_csv("train.csv", dtype=dtype_dictionary, 
                 keep_default_na=False, 
                 error_bad_lines=False,
                 na_values=['na',''])
```

**NOTE: **As NANs are considered to be float in pandas don’t forget to convert integer data_types to float if your columns contain NANs.

**5. Loading Data in Chunks:**

Memory [Issues](https://wesmckinney.com/blog/update-on-upcoming-pandas-v0-10-new-file-parser-other-performance-wins/) in pandas read_csv() are there for a long time. So one of the best workarounds to load large datasets is in chunks.

**Note:** loading data in chunks is actually slower than reading whole data directly as you need to concat the chunks again but you can load files with more than 10’s of GB’s easily.

<script src="https://gist.github.com/PrudhviVajja/8309cb7ca2e833b3401026acbd97cb2c.js"></script>

<iframe src="https://giphy.com/embed/3rgXBsmYd60rL3w7sc" width="480" height="240" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/nicolettegroome-animation-adventure-time-bmo-3rgXBsmYd60rL3w7sc">via GIPHY</a></p>

### 6. Multiprocessing using pandas:

As pandas don’t have njobs variable to make use of multiprocessing power. we can utilize multiprocessinglibrary to handle chunk size operations asynchronously on multi-threads which can reduce the run time by half.

<script src="https://gist.github.com/PrudhviVajja/b0c7f7aab2bfe6a261d6ec957a07f609.js"></script>

> **Note: **you need to define pool in __main__ only because only main can distribute pool asynchronously among multiple processers.

### 7. Dask Instead of Pandas:

Although [Dask](https://github.com/dask/dask) doesn’t provide a wide range of data preprocessing functions such as pandas it supports parallel computing and loads data faster than pandas

```python
import dask.dataframe as dd

data = dd.read_csv("train.csv",dtype={'MachineHoursCurrentMeter': 'float64'},assume_missing=True)
data.compute()
```

🔥**Pro Tip: **If you want to find the time taken by a jupyter cell to run just add %%time magic function at the start of the cell

Libraries to try out: [**Paratext](https://github.com/wiseio/paratext), [Datatable](https://github.com/h2oai/datatable).**
> Their’s is an another way, You can rent a VM in the cloud, with 64 cores and 432GB RAM, for ~$3/hour or even a better price with some googling.
> **caveat**: you need to spend the next week configuring it.

Thanks for reaching until the end, I hope you learned something new. Happy Loading….✌️. (👏 If you like it.)
> ### Comment below the tricks that you used to load your data faster I will add them to the list.

***References(Add them to your blog list):***

🔥[ Itamar Turner-Trauring](https://pythonspeed.com) — Speed Python Master (Must ✅).

🔥 [Gouthaman Balaraman](http://gouthamanbalaraman.com/) — quantitative finance with python (Must ✅).

Connect with me on [Linkedin](https://www.linkedin.com/in/prudhvi-vajja-22079610b/).
