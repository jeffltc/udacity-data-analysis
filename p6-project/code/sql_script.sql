sql

survive_rate_according_to_tags

select cast(count(1) as float) /(select count(*) from titanic_data_edit where Sex = 'male' ) as survive_rate
from titanic_data_edit
where Sex = 'male'  and survived = 1
group by survived;