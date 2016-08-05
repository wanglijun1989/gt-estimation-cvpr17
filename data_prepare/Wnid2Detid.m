function [wnid2detid] = Wnid2Detid(synsets)
wnid2detid = containers.Map();
for i = 1:length(synsets)
    wnid2detid(synsets(i).WNID) = synsets(i).ILSVRC2013_DET_ID;
end

end

