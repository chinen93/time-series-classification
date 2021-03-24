# Time series classification 

Implementation of Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline (2016, [arXiv](https://arxiv.org/abs/1611.06455)) in PyTorch.

## Use

Run `docker-compose build && docker-compose up` and open `localhost:8888` in your browser and open the `train.ipynb` notebook.

To use your own data, implement a Dataset class as in `src/project/data.py` and wrap that in a torch `DataLoader`.


# References

https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
https://github.com/hfawaz/dl-4-tsc
https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline
https://github.com/tcapelle/timeseries_fastai
