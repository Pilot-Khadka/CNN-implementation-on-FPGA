void convolution(int matrix [5][5], int kernel [3][3] ,int result[3][3])
{
	int i,j,m,n,ii,jj,mm,nn;
	float out;
	int rows=3,cols=3,kcols=3, krows=3;
	int kCenterX = kcols / 2;
	int kCenterY = krows / 2;

	for(i=0; i < rows; ++i)              // rows
	{
	    for(j=0; j < cols; ++j)          // columns
	    {
	        for(m=0; m < krows; ++m)     // kernel rows
	        {
	            mm = krows - 1 - m;      // row index of flipped kernel

	            for(n=0; n < kcols; ++n) // kernel columns
	            {
	                nn = kcols - 1 - n;  // column index of flipped kernel

	                // index of input signal, used for checking boundary
	                ii = i + (kCenterY - mm);
	                jj = j + (kCenterX - nn);

	                // ignore input samples which are out of bound
	                if( ii >= 0 && ii < rows && jj >= 0 && jj < cols )
	                    result[i][j] += matrix[ii][jj] * kernel[mm][nn];
	            }
	        }
	    }
	}
}
