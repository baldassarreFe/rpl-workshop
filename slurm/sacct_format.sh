# source this file in .bashrc so that the variable SACCT_FORMAT is exported:
# source sacct_format.sh

# Format field array, initially empty
FIELDS=()

# Comment/uncomment from these, use % to change the width of the field
FIELDS+=(jobid)
FIELDS+=(start)
# FIELDS+=(eligible)
# FIELDS+=(jobname)
# FIELDS+=(partition)
# FIELDS+=(maxvmsize)
# FIELDS+=(maxvmsizenode)
# FIELDS+=(maxvmsizetask)
# FIELDS+=(avevmsize)
FIELDS+=(maxrss)
# FIELDS+=(maxrssnode)
# FIELDS+=(maxrsstask)
# FIELDS+=(averss)
# FIELDS+=(maxpages)
# FIELDS+=(maxpagesnode)
# FIELDS+=(maxpagestask)
# FIELDS+=(avepages)
# FIELDS+=(mincpu)
# FIELDS+=(mincpunode)
# FIELDS+=(mincputask)
FIELDS+=(avecpu)
# FIELDS+=(ntasks)
FIELDS+=(alloccpus)
FIELDS+=(elapsed)
FIELDS+=(state%30)
FIELDS+=(exitcode)
# FIELDS+=(avecpufreq)
# FIELDS+=(reqcpufreqmin)
# FIELDS+=(reqcpufreqmax)
# FIELDS+=(reqcpufreqgov)
# FIELDS+=(consumedenergy)
# FIELDS+=(maxdiskread)
# FIELDS+=(maxdiskreadnode)
# FIELDS+=(maxdiskreadtask)
FIELDS+=(avediskread)
# FIELDS+=(maxdiskwrite)
# FIELDS+=(maxdiskwritenode)
# FIELDS+=(maxdiskwritetask)
FIELDS+=(avediskwrite)
FIELDS+=(allocgres)
FIELDS+=(nodelist)
# FIELDS+=(reqgres)

# Concatenate the fields into a comma-separated string
export SACCT_FORMAT="$(IFS=, ; echo "${FIELDS[*]}")"

# Example: show all jobs from your user since a certain date
# sacct -S 2019-10-30 -u "${USER}"
