/****** Script for SelectTopNRows command from SSMS  ******/
SELECT saleDate,convert(date,saledate) as DATE
from PortfolioProject.dbo.[Nashville  Housing]

Update PortfolioProject.dbo.[Nashville  Housing] 
set SaleDateConverted=convert(date,saledate)




alter table PortfolioProject.dbo.[Nashville  Housing]
add SaleDateConverted Date;

Update PortfolioProject.dbo.[Nashville  Housing]
set SaleDateConverted=convert(date,saledate)

SELECT SaleDateConverted
from PortfolioProject.dbo.[Nashville  Housing]

---------------------------------------------------------------

--property
select propertyAddress 
from PortfolioProject.dbo.[Nashville  Housing]

select *
from PortfolioProject.dbo.[Nashville  Housing]
--where propertyAddress is null
order by parcelID

select *
from PortfolioProject.dbo.[Nashville  Housing]
where propertyAddress is null
order by parcelID

select len(parcelID)
from PortfolioProject.dbo.[Nashville  Housing]
 
 select count(distinct parcelID ), count(parcelID)
 from PortfolioProject.dbo.[Nashville  Housing]
  

  select count(distinct propertyAddress), count(propertyAddress)
  from PortfolioProject.dbo.[Nashville  Housing]




-- Property adress is null for same parcel ID  so we can reuse that property address on same parcelID ( UniqueID is different)
select a.ParcelID,a.PropertyAddress,b.parcelID,b.PropertyAddress,isnull(a.propertyAddress,b.PropertyAddress)
from PortfolioProject.dbo.[Nashville  Housing] a 
join PortfolioProject.dbo.[Nashville  Housing] b
on a.parcelID=b.parcelID
and a.[UniqueID ]<>b.[UniqueID ]
where a.PropertyAddress is null


--update the table
update a
set PropertyAddress=isnull(a.propertyAddress,b.PropertyAddress)
from PortfolioProject.dbo.[Nashville  Housing] a 
join PortfolioProject.dbo.[Nashville  Housing] b
on a.parcelID=b.parcelID
and a.[UniqueID ]<>b.[UniqueID ]

--select *
--from PortfolioProject.dbo.[Nashville  Housing]
--where propertyAddress is null
--order by parcelID


-- looking for property address and property city


select propertyAddress 
from PortfolioProject.dbo.[Nashville  Housing]

select 
substring(propertyAddress,1,charindex(',',propertyAddress)-1) as Address,

substring(propertyAddress,charindex(',',propertyAddress)+1,len(propertyAddress)) as PropertyCity
from PortfolioProject.dbo.[Nashville  Housing]

alter table PortfolioProject.dbo.[Nashville  Housing]
add PropertyAdd NVARCHAR(255);

alter table PortfolioProject.dbo.[Nashville  Housing]
add PropertyCity NVARCHAR(255);
 
 update PortfolioProject.dbo.[Nashville  Housing]
 set PropertyAdd=substring(propertyAddress,1,charindex(',',propertyAddress)-1)

 update PortfolioProject.dbo.[Nashville  Housing]
 set PropertyCity=substring(propertyAddress,charindex(',',propertyAddress)+1,len(propertyAddress)) 

 select * 
 from PortfolioProject.dbo.[Nashville  Housing]

 --looking for Owner's city and owner's address


 select OwnerAddress 
 from PortfolioProject.dbo.[Nashville  Housing]
 where OwnerAddress is not null

 select 
 parsename(replace(OwnerAddress,',','.'),1),
 parsename(replace(OwnerAddress,',','.'),2),
 parsename(replace(OwnerAddress,',','.'),3)

 from PortfolioProject.dbo.[Nashville  Housing]
 where OwnerAddress is not null

  select 
 parsename(replace(OwnerAddress,',','.'),3),
 parsename(replace(OwnerAddress,',','.'),2),
 parsename(replace(OwnerAddress,',','.'),1)

 from PortfolioProject.dbo.[Nashville  Housing]


 alter table PortfolioProject.dbo.[Nashville  Housing]
 add OwnerADD NVARCHAR(255);
  
alter table PortfolioProject.dbo.[Nashville  Housing]
add OwnerCity NVARCHAR(255);

alter table PortfolioProject.dbo.[Nashville  Housing]
add OwnerState NVARCHAR(255);

update PortfolioProject.dbo.[Nashville  Housing]
set OwnerADD = parsename(replace(OwnerAddress,',','.'),3) 

update PortfolioProject.dbo.[Nashville  Housing]
set OwnerCity=parsename(replace(OwnerAddress,',','.'),2) 

update PortfolioProject.dbo.[Nashville  Housing]
set OwnerState=parsename(replace(OwnerAddress,',','.'),1) 


select *

from PortfolioProject.dbo.[Nashville  Housing]
where OwnerAddress is not null

-------------------------------------------------- 
--SoldAsVacant
select Distinct(soldasvacant),count(soldasvacant)
from PortfolioProject.dbo.[Nashville  Housing]
group by SoldAsVacant
order by 2


select SoldAsVacant,
case 
	when SoldAsVacant='Y' then 'Yes'
	When SoldAsVacant='N' then 'No'
	else SoldAsVacant
End
from PortfolioProject.dbo.[Nashville  Housing]


Update PortfolioProject.dbo.[Nashville  Housing]
set SoldAsVacant=
case 
	when SoldAsVacant='Y' then 'Yes'
	When SoldAsVacant='N' then 'No'
	else SoldAsVacant
End


------------------------------------------------
--Remove Duplicates
with RowCTE as (
select *,
Row_number() over (
	partition by parcelID,
				PropertyAddress,
				SaleDate,
				SalePrice,
				LegalReference
				order by UniqueID
				) row_num
from  PortfolioProject.dbo.[Nashville  Housing]
)
select *
from RowCTE
where row_num > 1
Order by propertyaddress
	

--------------------------------------------------
--Removed Unused columns

Select * 
from
PortfolioProject.dbo.[Nashville  Housing]

alter table PortfolioProject.dbo.[Nashville  Housing]
drop column SaleDate,PropertyAddress,OwnerAddress, TaxDistrict

