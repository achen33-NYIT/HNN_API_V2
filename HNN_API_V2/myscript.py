from pprint import pprint as pp

mylist = [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1]

mydict = {"balls": []}
myarr = mydict["balls"]


id = 0
for num in range(1):
       myvaldict = {}
       myarr.append(myvaldict)
       myvaldict["id"] = id
       myvaldict["values"] = []
       id += 1
       for num in mylist:
               myvaldict["values"].append({"val": str(num)})
               



pp(mydict)


# {
#     "balls": [
#         {
#             "id": "0",
#             "values": [{ "val": "0" }, { "val": "0" }, { "val": "1" }]
#         },

#         {
#             "id": "40",
#             "values": [{ "val": "1" }, { "val": "0" }, { "val": "1" }]
#         }
#     ]
# }
