### Problem

Sometimes a Storage Node Operator may encounter the `database disk image is malformed` error in their log. This could happen during unplanned shutdown or reboot. The error indicates that one or more of the sqlite3 databases may have become corrupted.

### Solution

```
sudo apt update && sudo apt install sqlite3 -y
```

perform the integrity check (perform for each database), for example for bandwidth.db:
```
sqlite3 /path/to/storage/bandwidth.db "PRAGMA integrity_check;"
```
If you see errors in the output, then the check did not pass. We will unload all uncorrupted data and then load it back. But this could sometimes fail, too. If no errors occur here, you can skip all the following steps and start the storagenode again.

If you were not lucky and the check failed, then please try to fix the corrupted database(s) as shown below.

Now run the following commands in the shell. 

```
cp /storage/bandwidth.db /storage/bandwidth.db.bak
sqlite3 /storage/bandwidth.db
```
You will see a prompt from sqlite3. Run this SQL script:
```
.mode insert
.output /storage/dump_all.sql
.dump
.exit
```
We will edit the SQL file dump_all.sql
```
cat /storage/dump_all.sql | grep -v TRANSACTION | grep -v ROLLBACK | grep -v COMMIT >/storage/dump_all_notrans.sql
```
Remove the corrupted database (make sure that you have a backup!)
```
rm /storage/bandwidth.db
```
Now we will load the unloaded data into the new database
```
sqlite3 /storage/bandwidth.db ".read /storage/dump_all_notrans.sql"
```
Check that the new database (bandwidth.db in our example) has a size larger than 0:
```
ls -al bag/bag.db3
```
Now you should be able to reindex successfully:

```
ros2 bag reindex bag/ sqlite3
```

