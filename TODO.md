1. WoE & IV computations - done
2. Accounting for neighbors and repeats - Vasily (neighbors - done, repeats 12 extra elements)
3. Spatially stratified cross-fold validation - Angel (done, splits don't match what's reported, not sure what they actually did)
4. Handling NaNs - Angel (done)
5. Merge above code to be able to train / test - (Vasily - merged neighbors function to .ipynb / IV seems to be very close to the ones in the paper)
6. Implement code that selects best threshold with cross validation - not needed in WOE
7. Implement code that evaluates WOE model on test set - done using AUC
8. House keeping - move more notebook functions to utility for reuse
9. Implement success rate AUC functionality / plot
10. Recreate baseline result in paper
11. Extend above to the "updated" WOE
