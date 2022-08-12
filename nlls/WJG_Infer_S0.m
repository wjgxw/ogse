%remove the IVIM effect
%angus 2021.3.5
function [Map] = WJG_Infer_S0(indata, bvalues, minimum)
 % bvalues should be larger than 200

rows = size(indata,1);
cols = size(indata,2);
N    = size(indata,3);
ly   = length(bvalues);
bvalues = reshape(bvalues, ly, 1);  % must be a columns vector

Map = zeros(rows,cols);       
indata = abs(indata);
 % Threshold images before fitting
thresh = sum( (abs(indata(:,:,:)) > minimum), 3) > (N-1);
inData = abs(indata);   % in case user has input complex data
for r= 1:rows
    for c=1:cols
        if thresh(r,c) 
            ydata = inData(r,c,:);
            ydata = log(reshape(ydata,N,1));
%             lfit = ydata\-bvalues; 
            [a,~] = polyfit(bvalues,ydata,1);
%             scatter(bvalues,ydata);hold on
%             x = min(bvalues):0.01:max(bvalues);
%             y = a(1)*x+a(2);
%             plot(x,y)
            Map(r,c,1) =  a(2);
        end
    end
end
Map = exp(Map);
 
