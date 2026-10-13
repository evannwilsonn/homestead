# Fetch both public datasets into data/raw (neither is committed to git).
#   julia --project=. scripts/00_download.jl
using Downloads
raw = joinpath(@__DIR__, "..", "data", "raw"); mkpath(raw)
files = [
    ("AmesHousing.txt", "http://jse.amstat.org/v19n3/decock/AmesHousing.txt"),                         # De Cock (2011), 2006–2010
    ("fhfa_hpi_metro.csv", "https://www.fhfa.gov/hpi/download/quarterly_datasets/hpi_at_metro.csv"),         # FHFA metro index
    ("ames_residential_sales_2020_2022.xlsx",
     "https://www.cityofames.org/files/assets/city/v/1/assessor/documents/sales/residential-sales.xlsx"), # City of Ames Assessor
]
for (name, url) in files
    path = joinpath(raw, name)
    isfile(path) ? println("have  ", name) : (Downloads.download(url, path); println("got   ", name))
end
