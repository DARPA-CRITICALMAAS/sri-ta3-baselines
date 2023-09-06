1. WoE & IV computations - done
2. Accounting for neighbors and repeats - Vasily (neighbors - done, repeats 12 extra elements)
3. Spatially stratified cross-fold validation - Angel (done, splits don't match what's reported, not sure what they actually did)
4. Handling NaNs - Angel (done)
5. Merge above code to be able to train / test - (Vasily - merged neighbors function to .ipynb / IV seems to be very close to the ones in the paper)
6. Implement code that selects best threshold with cross validation - not needed in WOE
7. Implement code that evaluates WOE model on test set - done using AUC
8. Move more notebook functions to utility for reuse - done
9. Implement success rate AUC - done
10. Recreate baseline result in paper - done / conversation with Lawley TBD
11. Recreate CD deposit/occurences experiment, in addition MVT - Vasily
11. Extend above to the "updated" WOE (done)
    a. Updating the input columns / data - Vasily & Angel (done)
    b. Category rebinning? - Angel (done)
12. Extend the above to the H2O model - gradient boosting machines with scikit-learn?
    a. ???
    b. ???
13. Begin familiarizing with EIS tools, other tools to massage, clean, georeference, etc. input data