function z = depth(vector)

    x = 1:168; y = 1:168;
    [X, Y] = meshgrid(x, y);
    z = g2s(squeeze(vector(1, :, :) ./ vector(3, :, :)), squeeze(vector(2, :, :) ./ vector(3, :, :)), x', y');
    h = fspecial('gaussian', 3, 1);
    z = imfilter(z, h);

end
