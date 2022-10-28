Select * 
From PortfolioProject..CovidDeaths
order by 3,4

--Select * 
--From PortfolioProject..CovidVaccinations
--order by 3,4
--Data Selection for further processing


Select location, date, total_cases, new_cases, total_deaths, population 
From PortfolioProject..CovidDeaths
order by 1,2


--Total cases vs Total deaths
--chances of dying

Select location, date, total_cases, total_deaths, (total_deaths/total_cases)*100 as DeathRate
From PortfolioProject..CovidDeaths
where location like '%india%'
order by 1,2

--Total cases vs Population
--Chances of affecting

Select location, date, population, total_cases, (total_cases/population)*100 as affected_rate
From PortfolioProject..CovidDeaths
--where location like '%india%'
order by 1,2

--Countries with highest infected rate

Select location,population, max(total_cases) as HighestInfectedcount, max((total_cases/population))*100 as PercentPopulationAffected
From PortfolioProject..CovidDeaths
--where location like '%india%'
group by  location,population
order by PercentPopulationAffected desc


--Countries with highest death count per population

Select location,population, max(cast(total_deaths as int)) as HighestDeathCount, max((total_deaths/population))*100 as PercentPopulationDead
From PortfolioProject..CovidDeaths
--where location like '%india%'
where continent is not null
group by  location,population

order by PercentPopulationDead desc


--Groupby Continent
Select continent, max(cast(total_deaths as int)) as TotalDeathCount
From PortfolioProject..CovidDeaths
--where location like '%india%'
where continent is not null
group by continent

order by  TotalDeathCount desc


--Global data
Select  sum(new_cases) as Total_cases, sum(cast(new_deaths as int)) as Total_deaths,  (sum(cast(new_deaths as int))/sum(new_cases))*100 as Death_percentage
from PortfolioProject..CovidDeaths
--where continent is null
--group by date
order by 1,2



--Joining Death table and Vaccination table
Select Dea.continent,Dea.location, Dea.date, Dea.population, Vac.new_vaccinations,
sum(convert(int,Vac.new_Vaccinations)) over(partition by Dea.location order by Dea.date,Dea.location ) as total_vaccinations 
from PortfolioProject..CovidDeaths  Dea
join PortfolioProject..CovidVaccinations  Vac
	on Dea.location = Vac.location
	and Dea.date = Vac.date
where Dea.continent is not null
and Dea.location like '%india%'

order by 2,3

--Total vaccinations and total deaths
--vaccinations started on  march 3 2020 in world
--vaccinations started on jan 16 2021 in india
--

Select Dea.continent,Dea.location,  Dea.date, Dea.population, Vac.new_vaccinations, Dea.new_cases, sum(convert(int,Dea.new_deaths)) over
(partition by Dea.location order by Dea.date, Dea.location )as total_deaths,
 sum(convert(int,Vac.new_Vaccinations)) over(partition by Dea.location order by Dea.date, Dea.location) as Total_vaccinations

from PortfolioProject..CovidDeaths  Dea
join PortfolioProject..CovidVaccinations Vac
	on Dea.location= Vac.location
	and Dea.date=Vac.date
where Dea.location like 'india'

and vac.new_vaccinations is not null
--and Dea.location like '%india'
group by Dea.location, Dea.continent, Dea.date,Dea.population , Vac.new_vaccinations,dea.new_deaths,dea.new_cases
order by 2,3 

-- Maximum daily cases and daily deaths globally  
Select continent, location, max(new_cases) as peak_cases, max(new_deaths) as peak_deaths 
from PortfolioProject..CovidDeaths
where continent is not null
Group by continent, location
order by continent

--  first death in india on march 11 2020
select location,date,continent,sum(convert(int,new_deaths)) 
over (partition by location order by date , location) as Total_deaths
from PortfolioProject..CovidDeaths
where location like '%india%'

-- first death in world on jan 23 2020
select location,date,continent,sum(convert(int,new_deaths)) 
over (partition by location order by date , location) as Total_deaths
from PortfolioProject..CovidDeaths
where location like 'world'

--first case in india on jan 30 2020
select location,date,continent,sum(convert(int,new_cases)) 
over (partition by location order by date , location) as Total_cases
from PortfolioProject..CovidDeaths
where location like '%india%' 



--first case in world on jan 23 2020
select location,date,continent,sum(convert(int,new_deaths)) 
over (partition by location order by date , location) as Total_cases
from PortfolioProject..CovidDeaths
where location like 'china'

select continent, date, location, sum(new_cases) as confirmed_cases 
from PortfolioProject..CovidDeaths
where continent is not null
group by continent,date,location
order by 1,2


--global records of newcases,new_deaths,totalcases,totaldeaths,
select dea.continent,dea.location,dea.population,dea.date,dea.new_cases,dea.total_cases,
dea.new_deaths,dea.total_deaths,vac.new_tests,vac.total_tests,
vac.new_vaccinations,vac.total_vaccinations,vac.people_vaccinated,vac.people_fully_vaccinated
from PortfolioProject..CovidDeaths dea
join PortfolioProject..CovidVaccinations vac
on dea.location=vac.location
and dea.date=vac.date

order by 2,3


select dea.continent,dea.location, dea.date,dea.population,vac.new_vaccinations, 
sum(convert(int,vac.new_vaccinations)) over (partition by dea.location order by dea.date,dea.location) as peoplerollvacciated 
--,(peoplerollvaccinated/population)*100 as percentpeoplevaccinated
from PortfolioProject..CovidDeaths dea
join PortfolioProject..CovidVaccinations vac
on dea.location=vac.location
and dea.date=vac.date

with populationVSvaccination (continent,location,date,population,new_vaccinations,peoplerollvaccinated)
as 
(
select dea.continent,dea.location, dea.date,dea.population,vac.new_vaccinations, 
sum(convert(int,vac.new_vaccinations)) over (partition by dea.location order by dea.location,dea.date) as peoplerollvaccinated 
--(peoplerollvaccinated/population)*100 as percentpeoplevaccinated
from PortfolioProject..CovidDeaths dea
join PortfolioProject..CovidVaccinations vac
on dea.location=vac.location
and dea.date=vac.date
where dea.continent is not null
) 
select *, (peoplerollvaccinated/population)*100 as percent_population_vaccinated
from populationVSvaccination


--creating view to store data  for future visualizations

create view populationVSvaccination as 
select dea.continent,dea.location, dea.date,dea.population,vac.new_vaccinations, 
sum(convert(int,vac.new_vaccinations)) over (partition by dea.location order by dea.location,dea.date) as peoplerollvaccinated 
--(peoplerollvaccinated/population)*100 as percentpeoplevaccinated
from PortfolioProject..CovidDeaths dea
join PortfolioProject..CovidVaccinations vac
on dea.location=vac.location
and dea.date=vac.date
where dea.continent is not null